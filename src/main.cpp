#include "Runner/CLIPAX650.hpp"
#include "Runner/CLIPOnnx.hpp"

#include "string_utility.hpp"
#include "cmdline.hpp"
#include "tableprinter.hpp"

int main(int argc, char *argv[])
{
    std::string image_src = "./test.jpg";
    std::string text_src = "a cat";
    std::string vocab_path = "./onnx_models/mobile_sam_decoder.onnx";
    std::string image_encoder_model_path = "./onnx_models/mobile_sam_encoder.onnx";
    std::string text_encoder_model_path = "./onnx_models/mobile_sam_encoder.onnx";
    std::string decoder_model_path = "./onnx_models/mobile_sam_decoder.onnx";

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("dec", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("image", 'i', "image file or folder(jpg png etc....)", true, image_src);
    cmd.add<std::string>("text", 't', "text or txt file", true, text_src);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);

    cmd.parse_check(argc, argv);

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
    decoder_model_path = cmd.get<std::string>("dec");

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
    mClip->load_decoder(decoder_model_path);
    mClip->load_tokenizer(vocab_path);

    image_src = cmd.get<std::string>("image");
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
    std::vector<std::vector<float>> text_features;
    mClip->encode(texts, text_features);

    std::vector<std::vector<float>> image_features;
    std::vector<std::string> image_paths;
    cv::Mat src = cv::imread(image_src);
    if (src.data)
    {
        std::vector<float> feat;
        mClip->encode(src, feat);
        image_features.push_back(feat);
        image_paths.push_back(image_src);
    }
    else
    {
        if (!string_utility<std::string>::ends_with(image_src, "/") &&
            !string_utility<std::string>::ends_with(image_src, "\\"))
        {
            image_src += "/";
        }
        std::vector<std::string> image_list;
        cv::glob(image_src + "*.*", image_list);

        for (size_t i = 0; i < image_list.size(); i++)
        {
            std::string image_path = image_list[i];
            src = cv::imread(image_path);
            if (!src.data)
                continue;
            std::vector<float> feat;
            mClip->encode(src, feat);
            image_features.push_back(feat);
            image_paths.push_back(image_path);
        }
    }

    std::vector<std::vector<float>> logits_per_image, logits_per_text;
    auto time_start = std::chrono::high_resolution_clock::now();
    mClip->decode(image_features, text_features, logits_per_image, logits_per_text);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "decode Inference Cost time : " << diff.count() << "s" << std::endl;

    printf("\n");
    if (texts.size() > 1)
    {
        printf("per image:\n");
        printf("%32s|", "image path\\text");
        for (size_t i = 0; i < texts.size(); i++)
        {
            printf("%32s|", texts[i].c_str());
        }
        printf("\n");
        for (size_t i = 0; i < logits_per_image.size(); i++)
        {
            printf("%32s|", image_paths[i].c_str());
            for (size_t j = 0; j < logits_per_image[i].size(); j++)
            {
                printf("%32.2f|", logits_per_image[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n");
    printf("per text:\n");
    printf("%32s|", "text\\image path");
    for (size_t i = 0; i < image_paths.size(); i++)
    {
        printf("%32s|", image_paths[i].c_str());
    }
    printf("\n");
    for (size_t i = 0; i < logits_per_text.size(); i++)
    {
        printf("%32s|", texts[i].c_str());
        for (size_t j = 0; j < logits_per_text[i].size(); j++)
        {
            printf("%32.2f|", logits_per_text[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}