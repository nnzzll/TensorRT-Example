#include "NvInfer.h"
#include <stdexcept>
#include <numeric>
namespace common
{
    struct Params
    {
        int32_t batchSize{1};
        int32_t dlaCore{-1};
        bool int8{false};
        bool fp16{false};
        short inputH{512};
        short inputW{512};
        short minHU{-44};
        short maxHU{1307};
        float mean{380.69367274};
        float std{286.26850487};

        std::string inputTensorName{"input"};
        std::string outputTensorName{"output"};
        std::string imagePath{"/data/CLINIC/image/dataset6_CLINIC_0001_data.nii.gz"};
        std::string enginePath{"UNet5.engine.fp16"};
    };
    
    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8:
            return 1;
        }
        throw std::runtime_error("Invalid DataType."); //<stdexcept>
        return 0;
    }

    inline int64_t volume(const nvinfer1::Dims &d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    template <typename A, typename B>
    inline A divUp(A x, B n) { return (x + n - 1) / n; }

    struct InferDeleter
    {
        template <typename T>
        void operator()(T *obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
}