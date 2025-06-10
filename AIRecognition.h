#ifndef AIRECOGNITION_H
#define AIRECOGNITION_H

#include <QWidget>
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QLabel>
#include <QPushButton>

namespace Ui {
class AIRecognition;
}

class AIRecognition : public QWidget
{
    Q_OBJECT

public:
    explicit AIRecognition(QWidget *parent = nullptr);
    ~AIRecognition();

private slots:
    void uploadImage();
    void startIdentify();
    void downloadReport();
    void clearImage();
    void showHelp();
    void returnToMain();
    void showPreviousImage();
    void showNextImage();
    void enlargeImage();

private:
    void loadImagesFromFolder();
    void displayCurrentImage();

    Ui::AIRecognition *ui;
    QString imageFolderPath; // 保存上传的图片文件夹路径
    QStringList imageFileList;
    int currentIndex = 0;

    QGraphicsScene *scene;
};

#endif // AIRECOGNITION_H



