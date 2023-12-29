#include "signal.h"

#include "Runner/CLIPAX650.hpp"
#include "Runner/CLIPOnnx.hpp"

#include "../ax-pipeline/examples/common/common_pipeline/common_pipeline.h"
#include "../ax-pipeline/examples/common/video_demux.hpp"

#include "string_utility.hpp"
#include "cmdline.hpp"

#include "ax_engine_api.h"

#if __cplusplus
extern "C"
{
#include "common_sys.h"
}
#endif

volatile int gLoopExit = 0;

void __sigExit(int iSigNo)
{
    ALOGI("quit the loop");
    gLoopExit = 1;
    sleep(1);
    return;
}
CLIP *gClip = nullptr;
std::vector<std::string> gTexts;
std::vector<std::vector<float>> gTextFeatures;
std::vector<std::vector<float>> gImageFeatures(1);
void ai_inference_func(pipeline_buffer_t *buff)
{
    if (!gClip)
    {
        return;
    }

    cv::Mat src = cv::Mat(buff->n_height, buff->n_width, CV_8UC3, buff->p_vir);

    gClip->encode(src, gImageFeatures[0]);

    std::vector<std::vector<float>> logits_per_image, logits_per_text;
    gClip->decode(gImageFeatures, gTextFeatures, logits_per_image, logits_per_text);

    for (size_t i = 0; i < gTexts.size(); i++)
    {
        printf("%8s|", gTexts[i].c_str());
    }
    printf("\n");
    for (size_t i = 0; i < logits_per_image.size(); i++)
    {
        for (size_t j = 0; j < logits_per_image[i].size(); j++)
        {
            printf("%8.2f|", logits_per_image[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void _demux_frame_callback(const void *buff, int len, void *reserve)
{
    if (len == 0)
    {
        pipeline_buffer_t end_buf = {0};
        user_input((pipeline_t *)reserve, 1, &end_buf);
        ALOGN("mp4 file decode finish,quit the demux loop");
    }
    pipeline_buffer_t buf_h26x = {0};
    buf_h26x.p_vir = (void *)buff;
    buf_h26x.n_size = len;
    user_input((pipeline_t *)reserve, 1, &buf_h26x);
    usleep(1 * 1000);
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    std::string video_src;
    std::string text_src;
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    std::string decoder_model_path;
    int language = 0;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("dec", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("video", 0, "video file(*.mp4 *.h264 etc....)", true, video_src);
    cmd.add<std::string>("text", 't', "text or txt file", true, text_src);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);
    cmd.add<int>("language", 'l', "language choose, 0:english 1:chinese", true, 0);

    cmd.parse_check(argc, argv);

#ifdef AXERA_TARGET_CHIP_AX620
    COMMON_SYS_POOL_CFG_T poolcfg[] = {
        {1920, 1088, 1920, AX_YUV420_SEMIPLANAR, 10},
    };
#elif defined(AXERA_TARGET_CHIP_AX650)
    COMMON_SYS_POOL_CFG_T poolcfg[] = {
        {1920, 1088, 1920, AX_FORMAT_YUV420_SEMIPLANAR, 20},
    };
#endif
    COMMON_SYS_ARGS_T tCommonArgs = {0};
    tCommonArgs.nPoolCfgCnt = 1;
    tCommonArgs.pPoolCfg = poolcfg;
    /*step 1:sys init*/
    int s32Ret = COMMON_SYS_Init(&tCommonArgs);
    if (s32Ret)
    {
        ALOGE("COMMON_SYS_Init failed,s32Ret:0x%x\n", s32Ret);
        return -1;
    }

    /*step 3:npu init*/
#ifdef AXERA_TARGET_CHIP_AX620
    AX_NPU_SDK_EX_ATTR_T sNpuAttr;
    sNpuAttr.eHardMode = AX_NPU_VIRTUAL_1_1;
    s32Ret = AX_NPU_SDK_EX_Init_with_attr(&sNpuAttr);
    if (0 != s32Ret)
    {
        ALOGE("AX_NPU_SDK_EX_Init_with_attr failed,s32Ret:0x%x\n", s32Ret);
        return -1;
    }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    s32Ret = AX_ENGINE_Init(&npu_attr);
    if (0 != s32Ret)
    {
        ALOGE("AX_ENGINE_Init 0x%x", s32Ret);
        return -1;
    }
#endif

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
    decoder_model_path = cmd.get<std::string>("dec");
    language = cmd.get<int>("language");

    std::shared_ptr<CLIP> mClip;
    if (string_utility<std::string>::ends_with(image_encoder_model_path, ".onnx"))
    {
        mClip.reset(new CLIPOnnx);
    }
    else if (string_utility<std::string>::ends_with(image_encoder_model_path, ".axmodel"))
    {
        mClip.reset(new CLIPAX650);
    }
    else
    {
        fprintf(stderr, "no impl for %s\n", image_encoder_model_path.c_str());
        return -1;
    }

    mClip->load_image_encoder(image_encoder_model_path);
    mClip->load_text_encoder(text_encoder_model_path);
    // mClip->load_decoder(decoder_model_path);
    mClip->load_tokenizer(vocab_path, language == 1);

    gClip = mClip.get();

    video_src = cmd.get<std::string>("video");
    text_src = cmd.get<std::string>("text");

    std::vector<std::string> texts;
    if (string_utility<std::string>::ends_with(text_src, ".txt"))
    {
        std::ifstream infile;
        infile.open(text_src);
        if (!infile.good())
        {
            ALOGE("");
            return -1;
        }

        std::string s;
        while (getline(infile, s))
        {
            texts.push_back(s);
        }
        infile.close();
    }
    else
    {
        texts.push_back(text_src);
    }
    gTexts = texts;
    std::vector<std::vector<float>> text_features;
    auto time_start = std::chrono::high_resolution_clock::now();
    mClip->encode(texts, text_features);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;
    gTextFeatures = text_features;

    pipeline_t pipe = {0};
    {
        pipeline_ivps_config_t &config = pipe.m_ivps_attr;
        config.n_ivps_grp = 0; // 重复的会创建失败
        config.n_ivps_fps = 30;
        config.n_ivps_width = 1280;
        config.n_ivps_height = 720;
        config.b_letterbox = 1;
        config.n_fifo_count = 1; // 如果想要拿到数据并输出到回调 就设为1~4
    }
    pipe.enable = 1;
    pipe.pipeid = 0;
    pipe.m_input_type = pi_vdec_h264;
    pipe.m_output_type = po_buff_rgb;
    pipe.n_loog_exit = 0;
    pipe.m_vdec_attr.n_vdec_grp = 0;
    pipe.output_func = ai_inference_func; // 图像输出的回调函数

    create_pipeline(&pipe);

    VideoDemux demux;
    demux.Open(video_src, 0, _demux_frame_callback, &pipe);
    while (!gLoopExit)
    {
        usleep(1000 * 1000);
    }
    demux.Stop();

    destory_pipeline(&pipe);

    ALOGI("end of the demo");
    return 0;
}