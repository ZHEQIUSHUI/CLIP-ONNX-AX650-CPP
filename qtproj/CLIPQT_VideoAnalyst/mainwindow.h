#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "clip/Runner/CLIPAX650.hpp"
#include "clip/Runner/CLIPOnnx.hpp"

#include "ax-pipeline/examples/common/common_pipeline/common_pipeline.h"
#include "ax-pipeline/examples/common/video_demux.hpp"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

struct model_info_t
{
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    int language;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(model_info_t *model_info,
               QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_btn_add_text_clicked();

    void on_btn_remove_text_clicked();

    void on_btn_select_video_clicked();

public:
    Ui::MainWindow *ui;

    std::shared_ptr<CLIP> mClip;

    pipeline_t pipes[2] = {0};
    VideoDemux demux;
private slots:
    void on_btn_save_text_clicked();

private slots:
    void on_btn_load_txt_clicked();

private slots:
    void on_bn_stop_clicked();
};
#endif // MAINWINDOW_H
