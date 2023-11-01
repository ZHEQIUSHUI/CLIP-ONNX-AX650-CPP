#ifndef MYQLABEL_H
#define MYQLABEL_H
#include <QLabel>
#include <QPainter>
#include <qimage.h>

#include "mutex"

class myQLabel : public QLabel
{
    Q_OBJECT
private:
    std::mutex mLock;
    QImage cur_image;

    static QPoint getWindowPoint(QSize window, QSize img, QPoint pt)
    {
        float letterbox_rows = window.height();
        float letterbox_cols = window.width();
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / img.height()) < (letterbox_cols * 1.0 / img.width()))
        {
            scale_letterbox = (float)letterbox_rows * 1.0f / (float)img.height();
        }
        else
        {
            scale_letterbox = (float)letterbox_cols * 1.0f / (float)img.width();
        }
        resize_cols = int(scale_letterbox * (float)img.width());
        resize_rows = int(scale_letterbox * (float)img.height());
        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;
        float ratio_x = (float)img.height() / resize_rows;
        float ratio_y = (float)img.width() / resize_cols;
        auto x0 = pt.x();
        auto y0 = pt.y();
        x0 = x0 / ratio_x + tmp_w;
        y0 = y0 / ratio_y + tmp_h;
        return QPoint(x0, y0);
    }

    static QRect getTargetRect(QSize targetsize, QImage img)
    {
        return QRect(QPoint(getWindowPoint(targetsize, img.size(), {0, 0})), QPoint(getWindowPoint(targetsize, img.size(), {img.width(), img.height()})));
    }

    void paintEvent(QPaintEvent *event) override
    {
        std::lock_guard<std::mutex> tlock(mLock);
        QPainter p(this);
        p.drawImage(getTargetRect(this->size(), cur_image), cur_image);
        p.end();
    }

 signals:
    void repaint_signal();

 protected slots:
    void repaint_slot()
    {
        repaint();
    }

public:
    myQLabel(QWidget *parent) : QLabel(parent)
    {
        connect(this,SIGNAL(repaint_signal()),this,SLOT(repaint_slot()));
    }

    QImage getCurrentImage()
    {
        return cur_image;
    }

    void SetImage(QImage img)
    {
        {
            std::lock_guard<std::mutex> tlock(mLock);
            cur_image = img.copy();
        }

        emit repaint_signal();
    }
};
#endif // MYQLABEL_H
