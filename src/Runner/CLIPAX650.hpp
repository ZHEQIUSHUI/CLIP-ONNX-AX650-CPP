#pragma once
#include "CLIP.hpp"
#include "ax_model_runner_ax650.hpp"

class CLIPAX650 : public CLIP
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;
    cv::Mat input;

public:
    bool load_image_encoder(std::string encoder_path) override
    {
        m_encoder.reset(new ax_runner_ax650);
        m_encoder->init(encoder_path.c_str());
        input_height = m_encoder->get_algo_height();
        input_width = m_encoder->get_algo_width();
        ALOGI("input size %d %d", input_height, input_width);
        input = cv::Mat(input_height, input_width, CV_8UC3, m_encoder->get_input(0).pVirAddr);

        LEN_IMAGE_FEATURE = m_encoder->get_output(0).vShape[1];
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
        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        memcpy(image_features.data(), m_encoder->get_output(0).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));
    }
};
