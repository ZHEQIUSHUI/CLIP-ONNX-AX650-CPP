#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "QLabel"
#include "QGridLayout"
#include "QLineEdit"
#include "QStandardItemModel"
#include "QFileDialog"

#include "myqlabel.h"

#include "clip/string_utility.hpp"
#include "clip/cqdm.h"

#include "ax_engine_api.h"
#if __cplusplus
extern "C"
{
#include "common_sys.h"
}
#endif

CLIP *gClip = nullptr;
std::vector<std::string> gTexts;
std::vector<std::vector<float>> gTextFeatures;
std::vector<std::vector<float>> gImageFeatures(1);
std::vector<QTableWidgetItem*> gUpdateScoreItem;
MainWindow *windows;

void display_func(pipeline_buffer_t *buff)
{
    if(windows)
    {
        QImage disp = QImage((uchar*)buff->p_vir,buff->n_width,buff->n_height,QImage::Format_BGR888);
        windows->ui->label->SetImage(disp);
    }
}


void ai_inference_func(pipeline_buffer_t *buff)
{
    if (gClip)
    {
        cv::Mat src = cv::Mat(buff->n_height, buff->n_width, CV_8UC3, buff->p_vir);

        gClip->encode(src, gImageFeatures[0]);

        std::vector<std::vector<float>> logits_per_image, logits_per_text;
        gClip->decode(gImageFeatures, gTextFeatures, logits_per_image, logits_per_text);

        if(gUpdateScoreItem.size() == gTexts.size() &&
                logits_per_image.size() > 0 &&
                logits_per_image[0].size() == gTexts.size())
        {
            for (size_t i = 0; i < gTexts.size(); i++)
            {
                gUpdateScoreItem[i]->setText(QString::number(logits_per_image[0][i],'f',2));
            }

            int max_id = -1;
            float max_val = -MAXFLOAT;
            for (size_t i = 0; i < gTexts.size(); i++)
            {
                if(logits_per_image[0][i] > max_val)
                {
                    max_val = logits_per_image[0][i];
                    max_id = i;
                }
            }

            QBrush brush(QColor(0,255,0));
            for (size_t i = 0; i < gTexts.size(); i++)
            {
                if(max_id==i)
                {
                    gUpdateScoreItem[i]->setBackground(brush);
                }
                else
                {
                    gUpdateScoreItem[i]->setBackground(QBrush());
                }
            }


        }


        //        for (size_t i = 0; i < gTexts.size(); i++)
        //        {
        //            printf("%8s|", gTexts[i].c_str());
        //        }
        //        printf("\n");
        //        for (size_t i = 0; i < logits_per_image.size(); i++)
        //        {
        //            for (size_t j = 0; j < logits_per_image[i].size(); j++)
        //            {
        //                printf("%8.2f|", logits_per_image[i][j]);
        //            }
        //            printf("\n");
        //        }
        //        printf("\n");
    }
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
    usleep(30 * 1000);
}

MainWindow::MainWindow(
        model_info_t *model_info,
        QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    if (model_info)
    {
        // init sys
        {
            COMMON_SYS_POOL_CFG_T poolcfg[] = {
                {1920, 1088, 1920, AX_FORMAT_YUV420_SEMIPLANAR, 20},
            };
            COMMON_SYS_ARGS_T tCommonArgs = {0};
            tCommonArgs.nPoolCfgCnt = 1;
            tCommonArgs.pPoolCfg = poolcfg;
            /*step 1:sys init*/
            int s32Ret = COMMON_SYS_Init(&tCommonArgs);
            if (s32Ret)
            {
                ALOGE("COMMON_SYS_Init failed,s32Ret:0x%x\n", s32Ret);
                return;
            }
            /*step 3:npu init*/
            AX_ENGINE_NPU_ATTR_T npu_attr;
            memset(&npu_attr, 0, sizeof(npu_attr));
            npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
            s32Ret = AX_ENGINE_Init(&npu_attr);
            if (0 != s32Ret)
            {
                ALOGE("AX_ENGINE_Init 0x%x", s32Ret);
                return;
            }
        }
        // init pipe
        {
            //display
            {
                auto&pipe = pipes[0];
                pipeline_ivps_config_t &config = pipe.m_ivps_attr;
                config.n_ivps_grp = 0; // 重复的会创建失败
                config.n_ivps_fps = 30;
                config.n_ivps_width = 1280;
                config.n_ivps_height = 720;
                config.b_letterbox = 1;
                config.n_fifo_count = 1; // 如果想要拿到数据并输出到回调 就设为1~4

                pipe.enable = 1;
                pipe.pipeid = 0;
                pipe.m_input_type = pi_vdec_h264;
                pipe.m_output_type = po_buff_rgb;
                pipe.n_loog_exit = 0;
                pipe.m_vdec_attr.n_vdec_grp = 0;
                pipe.output_func = display_func; // 图像输出的回调函数

                create_pipeline(&pipe);
            }
            //inference
            {
                auto&pipe = pipes[1];
                pipeline_ivps_config_t &config = pipe.m_ivps_attr;
                config.n_ivps_grp = 1; // 重复的会创建失败
                config.n_ivps_fps = 30;
                config.n_ivps_width = 640;
                config.n_ivps_height = 360;
                config.b_letterbox = 1;
                config.n_fifo_count = 1; // 如果想要拿到数据并输出到回调 就设为1~4

                pipe.enable = 1;
                pipe.pipeid = 1;
                pipe.m_input_type = pi_vdec_h264;
                pipe.m_output_type = po_buff_rgb;
                pipe.n_loog_exit = 0;
                pipe.m_vdec_attr.n_vdec_grp = 0;
                pipe.output_func = ai_inference_func; // 图像输出的回调函数

                create_pipeline(&pipe);
            }
        }

        // init model
        {
            if (string_utility<std::string>::ends_with(model_info->image_encoder_model_path, ".onnx"))
            {
                mClip.reset(new CLIPOnnx);
            }
            else if (string_utility<std::string>::ends_with(model_info->image_encoder_model_path, ".axmodel"))
            {
                mClip.reset(new CLIPAX650);
            }
            else
            {
                fprintf(stderr, "no impl for %s\n", model_info->image_encoder_model_path.c_str());
                return;
            }

            mClip->load_image_encoder(model_info->image_encoder_model_path);
            mClip->load_text_encoder(model_info->text_encoder_model_path);
            mClip->load_decoder(model_info->decoder_model_path);
            mClip->load_tokenizer(model_info->vocab_path, model_info->language == 1);
            gClip = mClip.get();
        }
    }

    ui->setupUi(this);
    windows = this;
}

MainWindow::~MainWindow()
{
    destory_pipeline(&pipes[0]);
    destory_pipeline(&pipes[1]);
    AX_ENGINE_Deinit();
    COMMON_SYS_DeInit();
    delete ui;
}

void MainWindow::on_btn_add_text_clicked()
{
    ui->tableWidget->insertRow(ui->tableWidget->rowCount());
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1,1,new QTableWidgetItem("0.0"));
}

void MainWindow::on_btn_remove_text_clicked()
{
    int curRow = ui->tableWidget->currentRow();
    if (curRow >= 0)
        ui->tableWidget->removeRow(curRow);
}

void MainWindow::on_btn_select_video_clicked()
{
    on_bn_stop_clicked();
    ui->tableWidget->setEnabled(false);
    ui->btn_add_text->setEnabled(false);
    ui->btn_remove_text->setEnabled(false);

    // get context from table
    std::vector<std::string> texts;
    std::vector<QTableWidgetItem*> empty_list;
    gUpdateScoreItem.clear();
    for (int i = 0; i < ui->tableWidget->rowCount(); i++)
    {
        auto item = ui->tableWidget->item(i, 0);
        if (item && !item->text().simplified().isEmpty())
        {
            texts.push_back(item->text().simplified().toStdString());
            auto score_item = ui->tableWidget->item(i, 1);
            gUpdateScoreItem.push_back(score_item);
            ALOGI("add text: %s", texts.back().c_str());
            fflush(stdout);
        }
        else
        {
            empty_list.push_back(item);
        }
    }

    gTexts = texts;
    std::vector<std::vector<float>> text_features;
    auto time_start = std::chrono::high_resolution_clock::now();
    mClip->encode(texts, text_features);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;
    gTextFeatures = text_features;

    auto filename = QFileDialog::getOpenFileName(this, "", "", "Video(*.h264 *.mp4)");
    if (filename.isEmpty())
    {
        on_bn_stop_clicked();
        return;
    }

    if (!demux.Open(filename.toStdString(), 1, _demux_frame_callback, &pipes[0]))
    {
        ALOGE("demux.Open video failed");
    }
}

void MainWindow::on_bn_stop_clicked()
{
    demux.Stop();
    ui->tableWidget->setEnabled(true);
    ui->btn_add_text->setEnabled(true);
    ui->btn_remove_text->setEnabled(true);

}

