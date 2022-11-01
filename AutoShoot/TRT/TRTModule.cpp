#include "TRTModule.h"
#include"../Logger/Logger.h"
#include <fstream>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fmt/format.h>
#include <fmt/color.h>
#include<algorithm>
#include<vector>

static inline int argmax(const float *ptr, int len) {
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

template<class F, class T, class ...Ts>
T reduce(F &&func, T x, Ts... xs) {
    if constexpr (sizeof...(Ts) > 0){
        return func(x, reduce(std::forward<F>(func), xs...));
    } else {
        return x;
    }
}

template<class T, class ...Ts>
T reduce_max(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::max(a, b);}, x, xs...);
}

template<class T, class ...Ts>
T reduce_min(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::min(a, b);}, x, xs...);
}

static inline float iou(const float pts1[10], const float pts2[10]) {
    cv::Rect2f bbox1, bbox2;
    bbox1.x = reduce_min(pts1[0], pts1[2], pts1[4], pts1[6]);
    bbox1.y = reduce_min(pts1[1], pts1[3], pts1[5], pts1[7]);
    bbox1.width = reduce_max(pts1[0], pts1[2], pts1[4], pts1[6]) - bbox1.x;
    bbox1.height = reduce_max(pts1[1], pts1[3], pts1[5], pts1[7]) - bbox1.y;

    bbox2.x = reduce_min(pts2[0], pts2[2], pts2[4], pts2[6]);
    bbox2.y = reduce_min(pts2[1], pts2[3], pts2[5], pts2[7]);
    bbox2.width = reduce_max(pts2[0], pts2[2], pts2[4], pts2[6]) - bbox2.x;
    bbox2.height = reduce_max(pts2[1], pts2[3], pts2[5], pts2[7]) - bbox2.y;

    cv::Rect And = bbox1 | bbox2;
    cv::Rect U = bbox1 & bbox2;

    return U.area()*1.0 / And.area();
}

TRTModule::TRTModule(const std::string &onnx_file) {
    Init(onnx_file);

    assert((input_Index = Engine->getBindingIndex("input")) == 0);
    assert((output_Index = Engine->getBindingIndex("output")) == 1);
    inputdims = Engine->getBindingDimensions(input_Index);
    std::cout << "[INFO]: input dims " << inputdims.d[0] << " " << inputdims.d[1] << " " << inputdims.d[2] << " " << inputdims.d[3] << std::endl;
    outpusdims = Engine->getBindingDimensions(output_Index);
    std::cout << "[INFO]: output dims "<< outpusdims.d[0] << " " << outpusdims.d[1] << " " << outpusdims.d[2] << std::endl;

    cudaMalloc(&device_buffer[input_Index], inputdims.d[0]*inputdims.d[1]*inputdims.d[2]*inputdims.d[3] * sizeof(float));
    host_buffer[input_Index]=malloc(inputdims.d[0]*inputdims.d[1]*inputdims.d[2]*inputdims.d[3] * sizeof(float));

    InputWrappers.emplace_back(inputdims.d[2], inputdims.d[3], CV_32FC1, host_buffer[input_Index]);
    InputWrappers.emplace_back(inputdims.d[2], inputdims.d[3], CV_32FC1, host_buffer[input_Index] + sizeof(float) * inputdims.d[2] * inputdims.d[3] );
    InputWrappers.emplace_back(inputdims.d[2], inputdims.d[3], CV_32FC1, host_buffer[input_Index] + 2 * sizeof(float) * inputdims.d[2] * inputdims.d[3]);

    cudaMalloc(&device_buffer[output_Index], outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2] * sizeof(float));


    cudaStreamCreate(&stream);
    output_buffer = new float[outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2]];
    assert(output_buffer != nullptr);

}

TRTModule::~TRTModule() {
    delete[] output_buffer;
    cudaStreamDestroy(stream);
    cudaFree(device_buffer[input_Index]);
    cudaFree(device_buffer[output_Index]);
    Engine->destroy();

}

std::vector<bbox_t> TRTModule::operator()(const cv::Mat &src,float conf_thres,float iou_thres) const{
    auto start = std::chrono::system_clock::now();
    cv::Mat x= doPicture(src);
    doInference();
    std::vector<bbox_t> rst;
    int nc=outpusdims.d[2]-15;

    rst.reserve(outpusdims.d[1]);
    std::vector<uint8_t> removed(outpusdims.d[1]);

    for (int i = 0; i < outpusdims.d[1]; i++) {
        auto *box_buffer = output_buffer + i * outpusdims.d[2];  // 20->23
        if(box_buffer[4]<conf_thres) continue;
        if(removed[i]) continue;

        bbox_t temp_box;
        for(int j=0;j<4;j++){
            temp_box.rect[j]=box_buffer[j];
        }
        temp_box.conf=box_buffer[4];
        for(int j=0;j<10;j++){
            temp_box.pts[j]=box_buffer[5+j];
        }
        temp_box.class_id= argmax(box_buffer+15,4);
        rst.emplace_back(temp_box);
        for(int j=i+1;j<outpusdims.d[1];j++){
            auto *box_buffer2 = output_buffer + j * outpusdims.d[2];  // 20->23
            if(box_buffer2[4]<conf_thres) continue;
            if(removed[j]) continue;
            float temppoint[10];
            for(int k=0;k<10;k++){
                temppoint[k]=box_buffer2[5+k];
            }
            if(iou(temp_box.pts,temppoint)>iou_thres) removed[j] = true;
        }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "[INFO]：Do All time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return rst;
}

void TRTModule::Init(const std::string &strModelName) {
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    Logger gLogger;
    if(!exists(strTrtName))
    {
        std::cout << "[INFO]：Loading onnx model..." <<std::endl;
        std::cout << "[INFO]: build engine from onnx" << std::endl;
        // Logger gLogger;
        IRuntime* m_CudaRuntime = createInferRuntime(gLogger);
        IBuilder* builder = createInferBuilder(gLogger);
        builder->setMaxBatchSize(1);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

        IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1ULL << 30);

        //启用 FP16 精度推理
        config->setFlag(BuilderFlag::kFP16);

        Engine = builder->buildEngineWithConfig(*network, *config);
        Context = Engine->createExecutionContext();

        IHostMemory *gieModelStream = Engine->serialize();
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(gieModelStream->size());
        memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
        serialize_output_stream.open(strTrtName);
        serialize_output_stream<<serialize_str;
        serialize_output_stream.close();

        size_t free, total;
        cuMemGetInfo(&free, &total);
        std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB" << std::endl;
        std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;

        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
    }
    else{
        std::cout << "[INFO]: build engine from cache" << std::endl;
        std::cout << "[INFO]：Loading trt model..." <<std::endl;

        IRuntime* runtime = createInferRuntime(gLogger);

        std::string cached_path = strTrtName;
        std::ifstream fin(cached_path);
        std::string cached_engine = "";
        while (fin.peek() != EOF){
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
        }
        fin.close();
        Engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        Context = Engine->createExecutionContext();
        runtime->destroy();
    }

}

bool TRTModule::exists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

cv::Mat TRTModule::doPicture(const cv::Mat &cInMat) const {
    cv::Mat x;
    cInMat.copyTo(x);
    //cv::cvtColor(cInMat, x, cv::COLOR_BGR2RGB);
    float fx = (float) cInMat.cols / inputdims.d[2], fy = (float) cInMat.rows / inputdims.d[3];

    if (cInMat.cols != inputdims.d[2] || cInMat.rows != inputdims.d[3]) {
        cv::resize(x, x, {inputdims.d[2], inputdims.d[3]},cv::INTER_AREA);
    }

    x.convertTo(x, CV_32FC3,1.0/255);
    cv::split(x,InputWrappers);
    return x;
}

void TRTModule::doInference() const {
    //inference
    auto start = std::chrono::system_clock::now();
    cudaMemcpyAsync(device_buffer[input_Index], host_buffer[input_Index], inputdims.d[0]*inputdims.d[1]*inputdims.d[2]*inputdims.d[3] * sizeof(float), cudaMemcpyHostToDevice, stream);
    Context->enqueueV2(device_buffer, stream, nullptr);
    cudaMemcpyAsync(output_buffer, device_buffer[output_Index], outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2] * sizeof(float), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::system_clock::now();
    std::cout << "[INFO]: inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}

















