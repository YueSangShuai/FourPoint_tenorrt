#include <iostream>
#include"opencv2/opencv.hpp"
#include "AutoShoot/TRT/TRTModule.h"

int main(){
//    TRTModule trt("/home/yuesang/Project/CLionProjects/FourPoint_tensorrt/AutoShoot/model/best.onnx");
    cv::Mat temp=cv::imread("/media/yuesang/G/Robotmaster/dataset/data/images/train/0dd119c24d073325871696e372069ea6.jpg");

    TRTModule model("../AutoShoot/model/best32.onnx");
    auto temp1=model(temp);

    std::cout<<temp1.size()<<std::endl;

//    cv::Mat frame;
//    cv::VideoCapture capture = cv::VideoCapture("/media/yuesang/G/Robotmaster/dataset/video/video/1.mp4");
//    auto start = std::chrono::system_clock::now();
//    int count=0;
//    while(capture.read(frame)){
//
//        model(frame);
//        count++;
//    }
//    auto end = std::chrono::system_clock::now();
//    std::cout<<"[INFO]: Average_time "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/count<<std::endl;
}
