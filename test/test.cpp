#include <iostream>
#include "trt_crnn_base.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

int main(int argc, char** argv){
    std::string ini_path="../config/trt_crnn.ini";
    std::string wts_path="../models/trt_crnn.onnx";
    std::string engine_path="../models/trt_crnn.engine";
    std::string img_path="../models/549.jpg";
    std::vector<int> box1{903,447,1107,447,1107,511,903,511};
    std::vector<int> box2{902,518,1118,518,1118,583,902,583};
    std::vector<int> box3{903,585,1122,585,1122,651,903,651};
    std::vector<std::vector<int>> boxes;
    boxes.push_back(box1);
    boxes.push_back(box2);
    boxes.push_back(box3);
    
    std::shared_ptr<TRTCRNNBase> model=CreateTRTCRNN();
    model->init(ini_path);
    model->load_model(wts_path,engine_path);
    cv::Mat img=cv::imread(img_path);
    std::vector<TRTCRNNResult> results=model->recognize(img,boxes);
    std::cout<<1;
    for(auto& one:results){
        for(int i=0;i<4;i++){
            std::cout<<one.box_coordinates[i]<<" ";
        }
        std::cout<<std::endl;
        std::cout<<one.info<<std::endl;
        std::cout<<one.label<<std::endl;
        std::cout<<one.score<<std::endl;
        std::cout<<"************"<<std::endl;
    }
       



}
