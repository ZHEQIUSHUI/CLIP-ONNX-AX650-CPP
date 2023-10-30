#pragma once
#include "CLIP.hpp"
#include "BaseRunner.hpp"

class CLIPOnnx : public CLIP
{
private:
    std::shared_ptr<BaseRunner> m_encoder;
    cv::Mat input;

public:
    bool load_image_encoder(std::string encoder_path) override
    {
        m_encoder = CreateRunner(RT_OnnxRunner);
        BaseConfig config;
        config.nthread = 8;
        config.onnx_model = encoder_path;
        m_encoder->load(config);

        input_width = m_encoder->getInputShape(0)[3];
        input_height = m_encoder->getInputShape(0)[2];
        ALOGI("input size %d %d", input_height, input_width);

        LEN_IMAGE_FEATURE = m_encoder->getOutputShape(0)[1];
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);
        image_features_input = std::vector<float>(1024 * LEN_IMAGE_FEATURE);
        return true;
    }
    void encode(cv::Mat image, std::vector<float> &image_features) override
    {
        if (!m_encoder.get())
        {
            ALOGE("encoder not init");
            return;
        }
        cv::resize(image, input, cv::Size(input_width, input_height));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        float *inputPtr = (float *)m_encoder->getInputPtr(0);

        uchar *img_data = input.data;

        int letterbox_cols = input_width;
        int letterbox_rows = input_height;
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = (float(img_data[in_index]) - _mean_val[c]) * _std_val[c];
                }
            }
        }

        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        memcpy(image_features.data(), m_encoder->getOutputPtr(0), LEN_IMAGE_FEATURE * sizeof(float));
    }
};
