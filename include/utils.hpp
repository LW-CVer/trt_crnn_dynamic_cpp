#ifndef TRTCRNN_COMMON_H_
#define TRTCRNN_COMMON_H_

#include <math.h>
#include <iostream>
#include <map>
#include <vector>
#include "opencv2/opencv.hpp"

const std::string alphabet = "-0123456789*.";

namespace trt_crnn {

std::vector<cv::Mat> CropPerspective(
    cv::Mat& image, std::vector<std::vector<int>>& boxes,
    std::vector<std::vector<int>>& choosed_boxes)
{  // boxes内部应该包含8个值，代表检测出的4个点
    std::vector<cv::Mat> results;
    for (auto& coordinates : boxes) {
        //将坐标映射到当前图片尺度
        if (coordinates.size() != 8) {
            continue;
        }
        choosed_boxes.push_back(coordinates);
        std::vector<cv::Point2f> src_points(4);
        std::vector<cv::Point2f> dst_points(4);
        for (int i = 0; i < 4; i++) {
            src_points[i] =
                cv::Point(coordinates[i * 2], coordinates[i * 2 + 1]);
        }
        cv::Point2f temp = src_points[2];
        src_points[2] = src_points[3];
        src_points[3] = temp;

        int box_w = sqrt(pow(coordinates[2] - coordinates[0], 2) +
                         pow(coordinates[3] - coordinates[1], 2));
        int box_h = sqrt(pow(coordinates[6] - coordinates[0], 2) +
                         pow(coordinates[7] - coordinates[1], 2));

        dst_points[0] = cv::Point(0, 0);

        dst_points[1] = cv::Point(box_w, 0);
        dst_points[2] = cv::Point(0, box_h);
        dst_points[3] = cv::Point(box_w, box_h);
        cv::Mat result_image = cv::Mat::zeros(box_h, box_w, CV_8UC3);
        cv::Mat warpmatrix =
            cv::getPerspectiveTransform(src_points, dst_points);  //获取透视变换
        warpPerspective(image, result_image, warpmatrix,
                        result_image.size());  //透视变换
        results.push_back(result_image);
    }

    return results;
}

std::string strDecode(std::vector<int>& preds, bool raw, std::string alphabet)
{
    std::string str;
    if (raw) {
        for (auto v : preds) {
            str.push_back(alphabet[v]);
        }
    } else {
        for (size_t i = 0; i < preds.size(); i++) {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i]))
                continue;
            str.push_back(alphabet[preds[i]]);
        }
    }
    return str;
}

}  // namespace trt_crnn
#endif
