#include "NvInfer.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include <memory>
#include <vector>
#include <assert.h>
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void *mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;

public:
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr)
    {
    }

    GenericBuffer(size_t size, nvinfer1::DataType type) : mSize(size), mCapacity(size), mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer &operator=(GenericBuffer &&buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }
    void *data() { return mBuffer; }
    const void *data() const { return mBuffer; }
    size_t size() const { return mSize; }
    size_t nbBytes() const { return this->size() * common::getElementSize(mType); }
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }
    void resize(const nvinfer1::Dims &dims) { return this->resize(common::volume(dims)); }
    ~GenericBuffer() { freeFn(mBuffer); }
};

class DeviceAllocator
{
public:
    bool operator()(void **ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};
class DeviceFree
{
public:
    bool operator()(void *ptr) const { cudaFree(ptr); }
};
class HostAllocator
{
public:
    bool operator()(void **ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void *ptr) const
    {
        free(ptr);
    }
};
using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

class BufferManager
{
private:
    nvinfer1::ICudaEngine *mEngine;
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers;
    std::vector<void *> mDeviceBindings;
    int mBatchSize;
    void *getBuffer(const bool isHost, const std::string &tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t &stream = 0)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            void *dstPtr = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
            const void *srcPtr = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
            const size_t byteSize = mManagedBuffers[i]->deviceBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
            {
                if (async)
                    cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
                else
                    cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
            }
        }
    }

public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    BufferManager(nvinfer1::ICudaEngine* engine, const int batchSize = 0, const nvinfer1::IExecutionContext *context = nullptr) : mEngine(engine), mBatchSize(batchSize)
    {
        // Full Dims implies no batch size.
        assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {

            auto dims = context ? context->getBindingDimensions(i) : mEngine->getBindingDimensions(i);
            size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            int vecDim = mEngine->getBindingVectorizedDim(i);
            if (vecDim != -1)
            {
                int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
                dims.d[vecDim] = common::divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= common::volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    std::vector<void *> &getDeviceBindings() { return mDeviceBindings; }
    const std::vector<void *> &getDeviceBindings() const { return mDeviceBindings; }
    void *getDeviceBuffer(const std::string &tensorName) const
    {
        return getBuffer(false, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void *getHostBuffer(const std::string &tensorName) const
    {
        return getBuffer(true, tensorName);
    }
    size_t size(const std::string &tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->hostBuffer.nbBytes();
    }
    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(const cudaStream_t &stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(const cudaStream_t &stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager() = default;
};