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
#include <QNetworkAccessManager>  // 添加网络支持
#include <QNetworkReply>


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

    // 添加网络响应槽函数
    void handleDetectionResponse(QNetworkReply *reply);
    void showDetectionResult(bool isFake, double fakeConf, double realConf);

private:
    void loadImagesFromFolder();
    void displayCurrentImage();

    Ui::AIRecognition *ui;
    QString imageFolderPath; // 保存上传的图片文件夹路径
    QStringList imageFileList;
    int currentIndex = 0;

    QGraphicsScene *scene;
    QNetworkAccessManager *networkManager;  // 网络管理器
    QMap<QString, QJsonObject> detectionResults;  // 存储检测结果

    QDialog *waitingDialog;
    QLabel *waitingLabel;
    QTimer *waitingTimer;
    int waitingDots;

    void showWaitingDialog();
    void updateWaitingText();
    void hideWaitingDialog();
    void cancelDetection();
    // 新增取消相关成员
    QNetworkReply *currentReply;     // 当前网络请求
    bool requestFinished;            // 请求完成标志
    bool cancelRequested;            // 取消请求标志
    QPushButton *cancelButton;       // 取消按钮
};

#endif // AIRECOGNITION_H



