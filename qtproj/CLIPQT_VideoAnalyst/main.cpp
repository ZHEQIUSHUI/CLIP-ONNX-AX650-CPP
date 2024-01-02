#include "mainwindow.h"

#include <QApplication>
#include "style/DarkStyle.h"

#include "clip/cmdline.hpp"

int main(int argc, char *argv[])
{
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    int language = 0;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);
    cmd.add<int>("language", 'l', "language choose, 0:english 1:chinese", true, 0);

    cmd.parse_check(argc, argv);

    model_info_t model_info;

    model_info.vocab_path = cmd.get<std::string>("vocab");
    model_info.image_encoder_model_path = cmd.get<std::string>("ienc");
    model_info.text_encoder_model_path = cmd.get<std::string>("tenc");
    model_info.language = cmd.get<int>("language");

//    model_info.vocab_path = "/root/CLIP-ONNX-AX650-CPP/cn_vocab.txt";
//    model_info.image_encoder_model_path = "/root/CLIP-ONNX-AX650-CPP/onnx_models/cn_clip_vitb16.axmodel";
//    model_info.text_encoder_model_path = "/root/CLIP-ONNX-AX650-CPP/onnx_models/vitb16.txt.fp32.onnx";
//    model_info.language = 1;

    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);

    MainWindow w(&model_info);
    // MainWindow w(0);
    w.show();
    return a.exec();
}
