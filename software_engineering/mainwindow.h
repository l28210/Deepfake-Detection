#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>
#include <QLabel>         // 用于添加显示文字的 QLabel
#include <QScrollArea>    // 可选，如果你要用类型明示指针

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // 添加服务器地址获取方法
    QString getServerAddress() const;

private slots:
    void openAiChangeFace();
    void openAIRecognition();
    void openMoreFunction();

private:
    Ui::MainWindow *ui;
    QSettings settings;  // 用于存储服务器地址
};

#endif // MAINWINDOW_H





