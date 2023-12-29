#include "Runner/CLIPAX650.hpp"
#include "Runner/CLIPOnnx.hpp"

#include "string_utility.hpp"
#include "cmdline.hpp"

bool _file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

bool _file_read(const std::string &path, std::vector<char> &data)
{
    std::fstream fs(path, std::ios::in | std::ios::binary);

    if (!fs.is_open())
    {
        return false;
    }

    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg = fs.tellg();

    auto file_size = static_cast<size_t>(fs_end - fs_beg);
    auto vector_size = data.size();

    data.reserve(vector_size + file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();

    return true;
}

bool _file_dump(const std::string &path, char *data, int size)
{
    std::fstream fs(path, std::ios::out | std::ios::binary);

    if (!fs.is_open() || fs.fail())
    {
        fprintf(stderr, "[ERR] cannot open file %s \n", path.c_str());
    }

    fs.write(data, size);

    return true;
}

void process_texts(std::shared_ptr<CLIP> &mClip, std::vector<std::string> &lines, std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features)
{
    for (size_t i = 0; i < lines.size(); i++)
    {
        auto &s = lines[i];
        std::vector<std::string> ctxs = string_utility<std::string>::split(s, ":");
        if (ctxs.size() == 1)
        {
            std::vector<std::vector<float>> tmp_text_features;

            auto time_start = std::chrono::high_resolution_clock::now();
            mClip->encode(ctxs, tmp_text_features);
            auto time_end = std::chrono::high_resolution_clock::now();
            ALOGI("encode text [%s] cost time : %f", ctxs[0].c_str(), std::chrono::duration<double>(time_end - time_start).count());

            texts.push_back(ctxs[0]);
            text_features.insert(text_features.end(), tmp_text_features.begin(), tmp_text_features.end());
        }
        else if (ctxs.size() == 2)
        {
            if (_file_exist(ctxs[1]))
            {
                std::vector<char> tmp_c_text_feature;
                _file_read(ctxs[1], tmp_c_text_feature);

                std::vector<float> tmp_text_features;
                tmp_text_features.resize(tmp_c_text_feature.size() / sizeof(float));
                memcpy(tmp_text_features.data(), tmp_c_text_feature.data(), tmp_c_text_feature.size());

                texts.push_back(ctxs[0]);
                text_features.push_back(tmp_text_features);
                ALOGI("read text feature [%s] from %s", ctxs[0].c_str(), ctxs[1].c_str());
            }
            else
            {
                std::vector<std::string> tmp_texts{ctxs[0]};
                std::vector<std::vector<float>> tmp_text_features;
                auto time_start = std::chrono::high_resolution_clock::now();
                mClip->encode(tmp_texts, tmp_text_features);
                auto time_end = std::chrono::high_resolution_clock::now();
                ALOGI("encode text [%s] cost time : %f", ctxs[0].c_str(), std::chrono::duration<double>(time_end - time_start).count());

                texts.push_back(ctxs[0]);
                text_features.insert(text_features.end(), tmp_text_features.begin(), tmp_text_features.end());

                _file_dump(ctxs[1], (char *)tmp_text_features[0].data(), tmp_text_features[0].size() * sizeof(float));
                ALOGI("write text feature [%s] to %s", ctxs[0].c_str(), ctxs[1].c_str());
            }
        }
        else
        {
            ALOGE("text format error, %s", s.c_str());
        }
        /* code */
    }
}

int main(int argc, char *argv[])
{
    std::string image_src;
    std::string text_src;
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    int language = 0;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", false, text_encoder_model_path);
    cmd.add<std::string>("image", 'i', "image file or folder(jpg png etc....)", true, image_src);
    cmd.add<std::string>("text", 't', "text or txt file", true, text_src);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);
    cmd.add<int>("language", 'l', "language choose, 0:english 1:chinese", true, 0);

    cmd.parse_check(argc, argv);

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
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
    if (!text_encoder_model_path.empty())
    {
        mClip->load_text_encoder(text_encoder_model_path);
    }
    else
    {
        ALOGI("if you dont want to load text encoder, the '--text' args must be set like '--text dog:dog.bin', or set a txt file with content like \n\n%s    dog:dog.bin\n    bird:bird.bin\n    cat:cat.bin\n", MACRO_RED);
        ALOGI("and make sure the '.bin' file exist\n");
    }

    mClip->load_tokenizer(vocab_path, language == 1);

    image_src = cmd.get<std::string>("image");
    text_src = cmd.get<std::string>("text");

    std::vector<std::string> lines, texts;
    std::vector<std::vector<float>> text_features;
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
            lines.push_back(s);
        }
        infile.close();
    }
    else
    {
        lines.push_back(text_src);
    }

    process_texts(mClip, lines, texts, text_features);

    // auto time_start = std::chrono::high_resolution_clock::now();
    // mClip->encode(texts, text_features);
    // auto time_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = time_end - time_start;
    // std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;

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
            auto time_start = std::chrono::high_resolution_clock::now();
            mClip->encode(src, feat);
            auto time_end = std::chrono::high_resolution_clock::now();
            auto diff = time_end - time_start;
            std::cout << "image encode cost time : " << std::chrono::duration<double>(diff).count() << "s" << std::endl;
            image_features.push_back(feat);
            image_paths.push_back(image_path);
        }
    }

    std::vector<std::vector<float>> logits_per_image, logits_per_text;
    auto time_start = std::chrono::high_resolution_clock::now();
    mClip->decode(image_features, text_features, logits_per_image, logits_per_text);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto diff = time_end - time_start;
    std::cout << "postprocess cost time : " << std::chrono::duration<double>(diff).count() << "s" << std::endl;

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