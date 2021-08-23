#ifndef __TRT_CRNN_HPP__
#define __TRT_CRNN_HPP__

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "opencv2/core/mat.hpp"
#include "trt_crnn_base.hpp"
using namespace nvinfer1;

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

class Logger_trt_crnn : public nvinfer1::ILogger
{
   public:
    Logger_trt_crnn(Severity severity = Severity::kINFO)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity)
            return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

class TRTCRNN : public TRTCRNNBase
{
   public:
    TRTCRNN();
    ~TRTCRNN() override;
    int init(const std::string& ini_path) override;
    int load_model(std::string& wts_file, std::string& engine_file) override;
    std::vector<TRTCRNNResult> recognize(
        cv::Mat& image, std::vector<std::vector<int>>& boxes) override;

   private:
    nvinfer1::IExecutionContext* m_context;
    nvinfer1::IRuntime* m_runtime;
    nvinfer1::ICudaEngine* m_engine;
    cudaStream_t m_stream;

    void* m_buffers[2];  //一个存输入两个存输出 GPU
    std::vector<TRTCRNNResult> m_recognition_results;

    int m_gpu_index;
    int m_classes_num;
    bool m_use_fp16;
    int m_max_batchsize;
    std::string m_alphabet;
    void doInference(float* input, float* output, int h_scale, int w_scale, int time_width);
};
#endif
