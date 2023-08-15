#pragma once
#include <map>
#include "vector"
#include <string>
#include "fstream"
#include "thread"

#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"

#include "sample_log.h"
#include "Tokenizer.hpp"

#define LEN_IMAGE_FEATURE 512
#define LEN_TEXT_FEATURE 77

struct CLIP_IMAG_FEATURE_T
{
    float feature[LEN_IMAGE_FEATURE];
};

struct CLIP_TEXT_FEATURE_T
{
    int feature[LEN_TEXT_FEATURE];
};

class CLIP
{
protected:
    std::string device{"cpu"};
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> DecoderSession;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    const char *DecoderInputNames[2]{"image_features", "text"},
        *DecoderOutputNames[2]{"logits_per_image", "logits_per_text"};
    float _mean_val[3] = {0.48145466f * 255.f, 0.4578275f * 255.f, 0.40821073f * 255.f};
    float _std_val[3] = {1 / (0.26862954f * 255.f), 1 / (0.26130258f * 255.f), 1 / (0.27577711f * 255.f)};
    Tokenizer tokenizer;

    std::vector<float> image_features_input = std::vector<float>(1024 * LEN_IMAGE_FEATURE);
    std::vector<int> text_features_input = std::vector<int>(1024 * LEN_TEXT_FEATURE);

public:
    bool load_tokenizer(std::string vocab_path)
    {
        return tokenizer.load_tokenize(vocab_path);
    }

    bool load_decoder(std::string decoder_path)
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP_DECODER");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        // 设置图像优化级别
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        DecoderSession.reset(new Ort::Session(env, decoder_path.c_str(), session_options));
        if (DecoderSession->GetInputCount() != 2 || DecoderSession->GetOutputCount() != 2)
        {
            ALOGE("Model not loaded (invalid input/output count)");
            return false;
        }
        return true;
    }

    virtual bool load_encoder(std::string encoder_path) = 0;
    virtual void encode(cv::Mat image, std::vector<float> &image_features) = 0;

    void encode(std::vector<std::string> &texts, std::vector<std::vector<int>> &feats)
    {
        feats.resize(texts.size());
        for (size_t i = 0; i < texts.size(); i++)
        {
            tokenizer.encode_text(texts[i], feats[i]);
        }
    }

    void decode(std::vector<CLIP_IMAG_FEATURE_T> &image_features, std::vector<CLIP_TEXT_FEATURE_T> &text_features,
                std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        if (image_features.size() * LEN_IMAGE_FEATURE > image_features_input.size())
        {
            image_features_input.resize(image_features.size() * LEN_IMAGE_FEATURE);
        }
        if (text_features.size() * LEN_IMAGE_FEATURE > text_features_input.size())
        {
            text_features_input.resize(text_features.size() * LEN_IMAGE_FEATURE);
        }

        memset(image_features_input.data(), 0, image_features_input.size() * sizeof(float));
        auto image_features_input_ptr = image_features_input.data();
        memcpy(image_features_input_ptr, image_features.data(), image_features.size() * sizeof(CLIP_IMAG_FEATURE_T));

        memset(text_features_input.data(), 0, text_features_input.size() * sizeof(int));
        auto text_features_input_ptr = text_features_input.data();
        memcpy(text_features_input_ptr, text_features.data(), text_features.size() * sizeof(CLIP_TEXT_FEATURE_T));

        std::vector<Ort::Value> inputTensors;

        std::vector<int64_t> image_features_shape = {(int64_t)image_features.size(), LEN_IMAGE_FEATURE};
        std::vector<int64_t> text_features_shape = {(int64_t)text_features.size(), LEN_TEXT_FEATURE};

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, image_features_input.data(), image_features_input.size(), image_features_shape.data(), image_features_shape.size()));
        inputTensors.push_back(Ort::Value::CreateTensor<int>(
            memory_info_handler, text_features_input.data(), text_features_input.size(), text_features_shape.data(), text_features_shape.size()));

        Ort::RunOptions runOptions;
        auto DecoderOutputTensors = DecoderSession->Run(runOptions, DecoderInputNames, inputTensors.data(),
                                                        inputTensors.size(), DecoderOutputNames, 2);

        auto &logits_per_image_output = DecoderOutputTensors[0];
        auto logits_per_image_ptr = logits_per_image_output.GetTensorMutableData<float>();
        auto logits_per_image_shape = logits_per_image_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_image.resize(logits_per_image_shape[0]);
        for (size_t i = 0; i < logits_per_image.size(); i++)
        {
            logits_per_image[i].resize(logits_per_image_shape[1]);
            memcpy(logits_per_image[i].data(), logits_per_image_ptr + i * logits_per_image_shape[1], logits_per_image_shape[1] * sizeof(float));
        }

        auto &logits_per_text_output = DecoderOutputTensors[1];
        auto logits_per_text_ptr = logits_per_text_output.GetTensorMutableData<float>();
        auto logits_per_text_shape = logits_per_text_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_text.resize(logits_per_text_shape[0]);
        for (size_t i = 0; i < logits_per_text.size(); i++)
        {
            logits_per_text[i].resize(logits_per_text_shape[1]);
            memcpy(logits_per_text[i].data(), logits_per_text_ptr + i * logits_per_text_shape[1], logits_per_text_shape[1] * sizeof(float));
        }
    }

    void decode(std::vector<std::vector<float>> &image_features, std::vector<std::vector<int>> &text_features,
                std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        if (image_features.size() * LEN_IMAGE_FEATURE > image_features_input.size())
        {
            image_features_input.resize(image_features.size() * LEN_IMAGE_FEATURE);
        }
        if (text_features.size() * LEN_IMAGE_FEATURE > text_features_input.size())
        {
            text_features_input.resize(text_features.size() * LEN_IMAGE_FEATURE);
        }

        memset(image_features_input.data(), 0, image_features_input.size() * sizeof(float));
        auto image_features_input_ptr = image_features_input.data();
        for (size_t i = 0; i < image_features.size(); i++)
        {
            if (image_features[i].size() != LEN_IMAGE_FEATURE)
            {
                ALOGW("image_features index %d ,not equal %d\n", i, LEN_IMAGE_FEATURE);
                continue;
            }
            memcpy(image_features_input_ptr + i * LEN_IMAGE_FEATURE, image_features[i].data(), LEN_IMAGE_FEATURE * sizeof(float));
        }

        memset(text_features_input.data(), 0, text_features_input.size() * sizeof(int));
        auto text_features_input_ptr = text_features_input.data();
        for (size_t i = 0; i < text_features.size(); i++)
        {
            if (text_features[i].size() > LEN_TEXT_FEATURE)
            {
                ALOGW("text_features index %d ,bigger than %d\n", i, LEN_TEXT_FEATURE);
                continue;
            }
            memcpy(text_features_input_ptr + i * LEN_TEXT_FEATURE, text_features[i].data(), text_features[i].size() * sizeof(int));
        }
        std::vector<Ort::Value> inputTensors;

        std::vector<int64_t> image_features_shape = {(int64_t)image_features.size(), LEN_IMAGE_FEATURE};
        std::vector<int64_t> text_features_shape = {(int64_t)text_features.size(), LEN_TEXT_FEATURE};

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, image_features_input.data(), image_features_input.size(), image_features_shape.data(), image_features_shape.size()));
        inputTensors.push_back(Ort::Value::CreateTensor<int>(
            memory_info_handler, text_features_input.data(), text_features_input.size(), text_features_shape.data(), text_features_shape.size()));

        Ort::RunOptions runOptions;
        auto DecoderOutputTensors = DecoderSession->Run(runOptions, DecoderInputNames, inputTensors.data(),
                                                        inputTensors.size(), DecoderOutputNames, 2);

        auto &logits_per_image_output = DecoderOutputTensors[0];
        auto logits_per_image_ptr = logits_per_image_output.GetTensorMutableData<float>();
        auto logits_per_image_shape = logits_per_image_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_image.resize(logits_per_image_shape[0]);
        for (size_t i = 0; i < logits_per_image.size(); i++)
        {
            logits_per_image[i].resize(logits_per_image_shape[1]);
            memcpy(logits_per_image[i].data(), logits_per_image_ptr + i * logits_per_image_shape[1], logits_per_image_shape[1] * sizeof(float));
        }

        auto &logits_per_text_output = DecoderOutputTensors[1];
        auto logits_per_text_ptr = logits_per_text_output.GetTensorMutableData<float>();
        auto logits_per_text_shape = logits_per_text_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_text.resize(logits_per_text_shape[0]);
        for (size_t i = 0; i < logits_per_text.size(); i++)
        {
            logits_per_text[i].resize(logits_per_text_shape[1]);
            memcpy(logits_per_text[i].data(), logits_per_text_ptr + i * logits_per_text_shape[1], logits_per_text_shape[1] * sizeof(float));
        }
    }

    void decode(std::vector<float> &image_features, std::vector<int> &text_features,
                std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        std::vector<Ort::Value> inputTensors;

        std::vector<int64_t> image_features_shape = {(int64_t)image_features.size() / LEN_IMAGE_FEATURE, LEN_IMAGE_FEATURE};
        std::vector<int64_t> text_features_shape = {(int64_t)text_features.size() / LEN_TEXT_FEATURE, LEN_TEXT_FEATURE};

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, image_features.data(), image_features_shape[0] * image_features_shape[1], image_features_shape.data(), image_features_shape.size()));
        inputTensors.push_back(Ort::Value::CreateTensor<int>(
            memory_info_handler, text_features.data(), text_features_shape[0] * text_features_shape[1], text_features_shape.data(), text_features_shape.size()));

        Ort::RunOptions runOptions;
        auto DecoderOutputTensors = DecoderSession->Run(runOptions, DecoderInputNames, inputTensors.data(),
                                                        inputTensors.size(), DecoderOutputNames, 2);

        auto &logits_per_image_output = DecoderOutputTensors[0];
        auto logits_per_image_ptr = logits_per_image_output.GetTensorMutableData<float>();
        auto logits_per_image_shape = logits_per_image_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_image.resize(logits_per_image_shape[0]);
        for (size_t i = 0; i < logits_per_image.size(); i++)
        {
            logits_per_image[i].resize(logits_per_image_shape[1]);
            memcpy(logits_per_image[i].data(), logits_per_image_ptr + i * logits_per_image_shape[1], logits_per_image_shape[1] * sizeof(float));
        }

        auto &logits_per_text_output = DecoderOutputTensors[1];
        auto logits_per_text_ptr = logits_per_text_output.GetTensorMutableData<float>();
        auto logits_per_text_shape = logits_per_text_output.GetTensorTypeAndShapeInfo().GetShape();
        logits_per_text.resize(logits_per_text_shape[0]);
        for (size_t i = 0; i < logits_per_text.size(); i++)
        {
            logits_per_text[i].resize(logits_per_text_shape[1]);
            memcpy(logits_per_text[i].data(), logits_per_text_ptr + i * logits_per_text_shape[1], logits_per_text_shape[1] * sizeof(float));
        }
    }
};
