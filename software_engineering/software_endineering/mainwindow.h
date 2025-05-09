#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStackedWidget>

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

public:
    enum PageIndex {
        PAGE_AI_DETECT = 0,
        PAGE_FACE_SWAP = 1,
        PAGE_MAIN_MENU = 2
    };

private:
    QStackedWidget *stackedWidget;
};

#endif // MAINWINDOW_H
