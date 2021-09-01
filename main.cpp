#include <iostream>
#include <memory>
#include <vector>
#include <vtk-9.0/vtkNew.h>
#include <vtk-9.0/vtkImageData.h>
#include <vtk-9.0/vtkNIFTIImageReader.h>
#include <vtk-9.0/vtkPNGWriter.h>
#include <vtk-9.0/vtkNIFTIImageWriter.h>

#include "buffers.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"



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
    vtkNew<vtkNIFTIImageReader> reader;
    vtkNew<vtkImageData> result;
    int nSlice = 0;
    int maxSlice;
    short minHU, maxHU;
    float mean, std;
    long int inputSize;
    std::vector<short> input;
    std::vector<uint8_t> output;

public:
    nvinfer1::ICudaEngine *mEngine;
    OnnxUNet(const common::Params &param) : mParam(param), mEngine(nullptr)
    {
        reader->SetFileName(param.imagePath.c_str());
        reader->Update();
        result->SetDimensions(reader->GetOutput()->GetDimensions());
        result->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
        result->SetSpacing(reader->GetOutput()->GetSpacing());
        result->SetOrigin(reader->GetOutput()->GetOrigin());
        maxSlice = reader->GetOutput()->GetDimensions()[2];
        minHU = param.minHU;
        maxHU = param.maxHU;
        std = param.std;
        mean = param.mean;
        inputSize = param.inputH * param.inputW;
        input.resize(inputSize);
        output.resize(inputSize);
    }
    bool deserialize();
    bool processInput(const BufferManager &buffers);
    bool verifyOutput(const BufferManager &buffers);
    bool infer();
    bool write();
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
    for (nSlice; nSlice < maxSlice; nSlice++)
    {
        if (!processInput(buffers))
            return false;
        buffers.copyInputToDevice();
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status)
            return false;
        buffers.copyOutputToHost();
        if (!verifyOutput(buffers))
            return false;
    }

    return true;
}

bool OnnxUNet::processInput(const BufferManager &buffers)
{
    short *pixel = static_cast<short *>(reader->GetOutput()->GetScalarPointer(0, 0, nSlice));
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(mParam.inputTensorName));
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = *pixel++;
        input[i] = input[i] > maxHU ? maxHU : (input[i] < minHU ? minHU : input[i]);
        hostDataBuffer[i] = float((input[i] - mean) / std);
        //std::cout << hostDataBuffer[i] << std::endl;
    }
    return true;
}

bool OnnxUNet::verifyOutput(const BufferManager &buffers)
{
    float *ptr = static_cast<float *>(buffers.getHostBuffer(mParam.outputTensorName));
    uint8_t *pixel = static_cast<uint8_t *>(result->GetScalarPointer(0, 0, nSlice));

    for (int i = 0; i < inputSize; i++)
    {
        ptr[i] = 1 / (1 + exp(-ptr[i]));
        output[i] = ptr[i] >= 0.5 ? 255 : 0;
        *pixel++ = output[i];
    }
    return true;
}

bool OnnxUNet::write()
{
    vtkNew<vtkNIFTIImageWriter> writer;
    writer->SetFileName("output.nii.gz");
    writer->SetInputData(result);
    writer->Update();
    writer->Write();
}

int main(int, char **)
{

    common::Params param;
    OnnxUNet UNet(param);
    UNet.deserialize();
    std::cout << UNet.infer() << std::endl;
    UNet.write();
}
