#include <iostream>
#include <vtk-9.0/vtkNew.h>
#include <vtk-9.0/vtkImageData.h>
#include <vtk-9.0/vtkNIFTIImageReader.h>
#include <vtk-9.0/vtkPNGWriter.h>
#include "NvOnnxParser.h"
#include "buffers.h"
#include "NvInfer.h"
#include <vector>
#include "cuda_runtime_api.h"

#include <memory>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) override
    {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

class OnnxUNet
{
    template <typename T>
    using UniquePtr = std::unique_ptr<T, common::InferDeleter>;

private:
    common::Params mParam;
    int nSlice = 100;

public:
    nvinfer1::ICudaEngine *mEngine;
    OnnxUNet(const common::Params &param) : mParam(param), mEngine(nullptr) {}
    bool deserialize();
    bool processInput(const BufferManager &buffers);
    bool verifyOutput(const BufferManager &buffers);
    bool infer();
};

bool OnnxUNet::deserialize()
{
    std::ifstream enginefile(mParam.enginePath, std::ios::binary);
    enginefile.seekg(0, enginefile.end);
    long int filesize = enginefile.tellg();
    enginefile.seekg(0, enginefile.beg);

    std::vector<char> engineData(filesize);
    enginefile.read(engineData.data(), filesize);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    mEngine = runtime->deserializeCudaEngine(engineData.data(), filesize);
    if (!mEngine)
        return false;
    return true;
}

bool OnnxUNet::infer()
{
    BufferManager buffers(mEngine);
    auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;
    if (!processInput(buffers))
        return false;
    buffers.copyInputToDevice();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
        return false;
    buffers.copyOutputToHost();
    if(!verifyOutput(buffers))
        return false;
    return true;
}

bool OnnxUNet::processInput(const BufferManager &buffers)
{
    vtkNew<vtkNIFTIImageReader> reader;
    reader->SetFileName(mParam.imagePath.c_str());
    reader->Update();
    short minHU = mParam.minHU;
    short maxHU = mParam.maxHU;
    const long int inputSize = mParam.inputH * mParam.inputW;
    std::vector<short> fileData(inputSize);
    short *pixel = static_cast<short *>(reader->GetOutput()->GetScalarPointer(0, 0, nSlice));
    for (int i = 0; i < inputSize; i++)
        fileData[i] = *pixel++;
    nSlice++;
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParam.inputTensorName));
    for (int i = 0; i < inputSize; i++)
    {
        fileData[i] = fileData[i] > maxHU ? maxHU : (fileData[i] < minHU ? minHU : fileData[i]);
        hostDataBuffer[i] = float((fileData[i] - mParam.mean) / mParam.std);
        //std::cout << hostDataBuffer[i] << std::endl;
    }
    return true;
}

bool OnnxUNet::verifyOutput(const BufferManager &buffers)
{
    const long int outputSize = mParam.inputH*mParam.inputW;
    float*ptr = static_cast<float*>(buffers.getHostBuffer(mParam.outputTensorName));
    uint8_t output[outputSize];
    vtkNew<vtkImageData>mask;
    mask->SetDimensions(512,512,1);
    mask->AllocateScalars(VTK_UNSIGNED_CHAR,1);
    uint8_t* pixel = static_cast<uint8_t*>(mask->GetScalarPointer(0,0,0));

    for (int i=0;i<outputSize;i++)
    {
        ptr[i] = 1/(1+exp(-ptr[i]));
        output[i] = ptr[i]>=0.5?255:0;
        *pixel++ = output[i];
    }

    vtkNew<vtkPNGWriter>writer;
    writer->SetFileName("output.png");
    writer->SetInputData(mask);
    writer->Update();
    writer->Write();
    return true;
}

int main(int, char **)
{

    common::Params param;
    OnnxUNet UNet(param);
    UNet.deserialize();
    std::cout<<UNet.infer()<<std::endl;
}
