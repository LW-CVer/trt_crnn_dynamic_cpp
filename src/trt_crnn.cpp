#include "../include/trt_crnn.hpp"
#include <math.h>
#include <fstream>
#include "../include/INIReader.hpp"
#include "../include/ini.hpp"
#include "../include/utils.hpp"

TRTCRNN::TRTCRNN()
    : m_buffers{nullptr, nullptr}, m_context(nullptr), m_runtime(nullptr),
      m_engine(nullptr)
{
}

TRTCRNN::~TRTCRNN()
{
    for (int i = 0; i < 2; i++) {
        if (m_buffers[i] != nullptr) {
            cudaFree(m_buffers[i]);
            m_buffers[i] = nullptr;
        }
    }
}

int TRTCRNN::init(const std::string& ini_path)
{
    INIReader reader(ini_path);
    m_gpu_index = reader.GetInteger("device", "gpu_index", 0);
    m_use_fp16 = reader.GetBoolean("tensorrt", "fp16", false);
    m_max_batchsize = reader.GetInteger("tensorrt", "max_batchsize", 1);
    m_alphabet = reader.GetString("tensorrt", "alphabet", "-");
    //std::cout<<m_alphabet<<std::endl;
    m_classes_num = m_alphabet.length();
    //std::cout<<m_classes_num<<std::endl;

    return 0;
}

int TRTCRNN::load_model(std::string& onnx_file, std::string& engine_file)
{
    Logger_trt_crnn gLogger;
    cudaSetDevice(m_gpu_index);
    std::ifstream intrt(engine_file, std::ios::binary);
    if (intrt) {
        std::cout << "load local engine..." << engine_file << std::endl;
        m_runtime = nvinfer1::createInferRuntime(gLogger);
        intrt.seekg(0, std::ios::end);
        size_t length = intrt.tellg();
        intrt.seekg(0, std::ios::beg);
        std::vector<char> data(length);
        intrt.read(data.data(), length);
        m_engine = m_runtime->deserializeCudaEngine(data.data(), length);
        std::cout << "engine loaded." << std::endl;

    } else {
        std::cout << "create engine from onnx..." << std::endl;
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            return -1;
        }

        auto network = builder->createNetworkV2(
            1U << static_cast<int>(
                NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        if (!network) {
            return -1;
        }
        auto config = builder->createBuilderConfig();
        if (!config) {
            return -1;
        }

        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            return -1;
        }

        auto parsed = parser->parseFromFile(
            onnx_file.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        if (!parsed) {
            std::cout << "parse onnx failed." << std::endl;
            return -1;
        }

        nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
        nvinfer1::Dims output_dims_1 = network->getOutput(0)->getDimensions();
        /*std::cout << input_dims.d[0] << " " << input_dims.d[1] << " "
                  << input_dims.d[2] << " " << input_dims.d[3] << std::endl;
        std::cout << output_dims_1.d[0] << " " << output_dims_1.d[1] << " "
                  << output_dims_1.d[2] << " " << output_dims_1.d[3]
                  << std::endl;*/

        config->setAvgTimingIterations(1);
        config->setMinTimingIterations(1);
        config->setMaxWorkspaceSize(1 << 20);
        auto input = network->getInput(0);
        //设置动态维度
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{1, 1, 32, 63});
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{1, 1, 32, 128});
        profile->setDimensions(input->getName(),
                               nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{1, 1, 32, 960});
        config->addOptimizationProfile(profile);

        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(1 << 30);

        if (m_use_fp16) {
            config->setFlag(BuilderFlag::kFP16);
        }

        m_engine = builder->buildEngineWithConfig(*network, *config);
        nvinfer1::IHostMemory* engine_serialize = m_engine->serialize();
        std::ofstream out(engine_file.c_str(), std::ios::binary);
        out.write((char*)engine_serialize->data(), engine_serialize->size());

        std::cout << "serialize the engine to " << engine_file << std::endl;

        engine_serialize->destroy();
        parser->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();
    }
    m_context = m_engine->createExecutionContext();
    // m_engine->destroy();context还要使用时，对应的engine不能被destroy（）
    if (m_context == nullptr) {
        return -1;
    }
    /*
    for (int b = 0; b < m_engine->getNbBindings(); ++b) {
        if (m_engine->bindingIsInput(b))

            std::cout << "input:" << b << std::endl;
        else
            std::cout << "output:" << b << std::endl;
    }*/

    cudaStreamCreate(&m_stream);

    std::cout << "RT init done!" << std::endl;
    return 0;
}

void TRTCRNN::doInference(float* input, float* output, int h_scale, int w_scale,
                          int time_width)
{
    // const ICudaEngine& engine = m_context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.

    // assert(engine.getNbBindings() == 2);
    // std::cout << 1.1 << std::endl;

    // In order to bind the buffers, we need to know the names of the input and
    // output tensors. Note that indices are guaranteed to be less than

    // IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(m_INPUT_BLOB_NAME);
    // const int outputIndex = engine.getBindingIndex(m_OUTPUT_BLOB_NAME);
    // std::cout << h_scale << " " << w_scale << std::endl;

    // m_context->setBindingDimensions(0, nvinfer1::Dims3{1, h_scale, w_scale});
    // Create GPU buffers on device
    nvinfer1::Dims4 input_shape{1, 1, h_scale, w_scale};
    m_context->setBindingDimensions(0, input_shape);

    CHECK(cudaMalloc(&m_buffers[0], 1 * h_scale * w_scale * sizeof(float)));
    CHECK(
        cudaMalloc(&m_buffers[1], time_width * m_classes_num * sizeof(float)));

    // DMA input batch data to device, infer on the batch asynchronously, and
    // DMA output back to host
    CHECK(cudaMemcpyAsync(m_buffers[0], input,
                          1 * h_scale * w_scale * sizeof(float),
                          cudaMemcpyHostToDevice, m_stream));
    m_context->enqueueV2(m_buffers, m_stream, nullptr);
    // m_context->enqueue(1, m_buffers, m_stream, nullptr);
    CHECK(cudaMemcpyAsync(output, m_buffers[1],
                          time_width * m_classes_num * sizeof(float),
                          cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);

    CHECK(cudaFree(m_buffers[0]));
    CHECK(cudaFree(m_buffers[1]));
}

std::vector<TRTCRNNResult> TRTCRNN::recognize(
    cv::Mat& image, std::vector<std::vector<int>>& boxes)
{
    m_recognition_results.clear();
    std::vector<std::vector<int>> choosed_boxes;
    std::vector<cv::Mat> new_images =
        trt_crnn::CropPerspective(image, boxes, choosed_boxes);
    // image的h必须等于32
    int index = 0;
    for (auto& new_image : new_images) {

        int new_width = (int)new_image.cols / (new_image.rows / 32.0);

        cv::resize(new_image, new_image, cv::Size(new_width, 32));

        // cv::imwrite("./"+std::to_string(index)+".jpg",new_image);
        cv::cvtColor(new_image, new_image, cv::COLOR_BGR2GRAY);
        // auto start1 = std::chrono::system_clock::now();
        // auto start2 = std::chrono::system_clock::now();
        int time_width = new_image.cols / 4 + 1;
        float* input = new float[new_image.rows * new_image.cols * 1];
        float* output = new float[time_width * m_classes_num];
        // auto start3 = std::chrono::system_clock::now();
  
        float* input_ptr_pos = input;
        for (int i = 0; i < new_image.rows; i++) {
            for (int j = 0; j < new_image.cols; j++) {
                *input_ptr_pos =
                    (((float)new_image.at<uchar>(i, j)) / 255 - 0.5) / 0.5;
                input_ptr_pos++;
            }
        }
    
        doInference(input, output, new_image.rows, new_image.cols, time_width);
        
        std::vector<int> preds;
        for (int i = 0; i < time_width; i++) {
            int maxj = 0;
            for (int j = 1; j < m_classes_num; j++) {
                if (output[m_classes_num * i + j] >
                    output[m_classes_num * i + maxj])
                    maxj = j;
            }
            preds.push_back(maxj);
        }

        TRTCRNNResult temp_result;
        temp_result.label = 0;
        temp_result.score = 0.99;
        temp_result.info = trt_crnn::strDecode(preds, false, m_alphabet);
        //使用检测坐标的外接矩形
        temp_result.box_coordinates[0] =
            choosed_boxes[index][0] < choosed_boxes[index][6]
                ? choosed_boxes[index][0]
                : choosed_boxes[index][6];
        temp_result.box_coordinates[1] =
            choosed_boxes[index][1] < choosed_boxes[index][3]
                ? choosed_boxes[index][1]
                : choosed_boxes[index][3];
        temp_result.box_coordinates[2] =
            choosed_boxes[index][4] > choosed_boxes[index][2]
                ? choosed_boxes[index][4]
                : choosed_boxes[index][2];
        temp_result.box_coordinates[3] =
            choosed_boxes[index][5] > choosed_boxes[index][7]
                ? choosed_boxes[index][5]
                : choosed_boxes[index][7];

        m_recognition_results.push_back(temp_result);
        index++;
        delete input;
        delete output;
    }
    return m_recognition_results;
}

std::shared_ptr<TRTCRNNBase> CreateTRTCRNN()
{
    return std::shared_ptr<TRTCRNNBase>(new TRTCRNN());
}
