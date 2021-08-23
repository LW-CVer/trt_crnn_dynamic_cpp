#ifndef __TRT_CRNN_BASE_HPP__
#define __TRT_CRNN_BASE_HPP__
#include <memory>
#include <string>
#include <vector>
#include "opencv2/core/mat.hpp"
struct TRTCRNNResult
{
    int label;
    int box_coordinates[4];
    float score;
    std::string info;
};

class TRTCRNNBase
{
   public:
    virtual ~TRTCRNNBase() = default;
    virtual int init(const std::string& ini_path) = 0;
    virtual int load_model(std::string& onnx_file,
                           std::string& engine_file) = 0;
    virtual std::vector<TRTCRNNResult> recognize(cv::Mat& image,std::vector<std::vector<int>>& boxes) = 0;
};

std::shared_ptr<TRTCRNNBase> CreateTRTCRNN();
#endif
