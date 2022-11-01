#ifndef FOURPOINT_TENSORRT_TRTMODULE_H
#define FOURPOINT_TENSORRT_TRTMODULE_H

#include <opencv2/core.hpp>
#include <NvInfer.h>
#include "../Logger/Logger.h"



struct alignas(float)  bbox_t {
    float rect[4]; //xywh
    float conf; //置信度
    float pts[10]; // 四个点坐标
    float class_id;//类别

    bool operator==(const bbox_t&) const = default;
    bool operator!=(const bbox_t&) const = default;
};

class TRTModule {
public:
    explicit TRTModule(const std::string &onnx_file);

    ~TRTModule();

    TRTModule(const TRTModule &) = delete;

    std::vector<bbox_t> operator()(const cv::Mat &src,float conf_thres=0.5,float iou_thres=0.5) const;



private:
    cv::Mat doPicture(const cv::Mat& cInMat) const;

    void doInference() const;

    void Init(const std::string &strModelName);

    bool exists(const std::string &name);

private:

    nvinfer1::ICudaEngine *Engine;
    nvinfer1::IExecutionContext *Context;
    Logger gLogger;
    int input_Index,output_Index;

    Dims inputdims,outpusdims;
    mutable void *device_buffer[2];
    mutable void *host_buffer[2];
    std::vector<cv::Mat> InputWrappers;
    cudaStream_t stream;
    float* output_buffer;
private:

};

#endif //FOURPOINT_TENSORRT_TRTMODULE_H
