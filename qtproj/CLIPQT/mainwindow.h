#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "clip/Runner/CLIPAX650.hpp"
#include "clip/Runner/CLIPOnnx.hpp"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(std::string image_src,
               std::string vocab_path,
               std::string image_encoder_model_path,
               std::string text_encoder_model_path,
               int language,
               QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btn_search_clicked();

    void on_txt_context_returnPressed();

private:
    Ui::MainWindow *ui;

    int current_row = 0;
    int current_col = 0;
    // int max_row = 5;
    int max_display = 20;
    void add_image_text_label(QString image_path = QString(), QString text = QString(), int max_row = 5);
    void clear_image_text_label();

    std::shared_ptr<CLIP> mClip;
    std::vector<std::vector<float>> image_features;
    std::vector<std::string> image_paths;
};
#endif // MAINWINDOW_H
