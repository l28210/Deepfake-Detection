// customgraphicsview.h
#ifndef CUSTOMGRAPHICSVIEW_H
#define CUSTOMGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QMouseEvent>

class CustomGraphicsView : public QGraphicsView {
    Q_OBJECT
public:
    using QGraphicsView::QGraphicsView; // 继承构造函数

signals:
    void showlargergraph(); // 自定义信号

protected:
    void mouseDoubleClickEvent(QMouseEvent *event) override {
        Q_UNUSED(event); // 不使用 event 参数时避免警告
        emit showlargergraph(); // 发射信号
    }
};

#endif // CUSTOMGRAPHICSVIEW_H
