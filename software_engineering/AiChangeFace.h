#ifndef AICHANGEFACE_H
#define AICHANGEFACE_H

#include <QWidget>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QNetworkAccessManager>  // 添加网络支持
#include <QNetworkReply>
#include <QEventLoop>
#include <QLabel>
#include <QPushButton>

namespace Ui {
class AiChangeFace;
}

class AiChangeFace : public QWidget
{
    Q_OBJECT

public:
    explicit AiChangeFace(QWidget *parent = nullptr);
    ~AiChangeFace();

private slots:
    void returnToMain();            // 返回主界面
    void uploadMainImage();         // 上传主图
    void uploadReferenceImage();    // 上传参考图
    void startGenerating();         // 开始生成
    void clearImages();             // 清空图片
    void downloadImage();           // 下载图片
    void showHelp();                // 显示帮助

    // 添加网络响应槽函数
    void handleFaceSwapResponse(QNetworkReply *reply);

private:
    Ui::AiChangeFace *ui;
    bool mainImageUploaded = false;     // 主图上传状态
    bool referenceImageUploaded = false; // 参考图上传状态
    bool imageGenerated = false;         // 图片生成状态
    QString generatedImagePath;          // 生成的图片路径
    QByteArray sourceImageData;          // 源图片数据
    QByteArray targetImageData;          // 目标图片数据

    // 用于显示图片的场景
    QGraphicsScene *mainScene;
    QGraphicsScene *referenceScene;
    QGraphicsScene *generatedScene;

    // 图片项
    QGraphicsPixmapItem *mainPixmapItem;
    QGraphicsPixmapItem *referencePixmapItem;
    QGraphicsPixmapItem *generatedPixmapItem;

    // 网络管理器
    QNetworkAccessManager *networkManager;

private:
    QNetworkReply *currentReply;  // 添加这行
    QDialog *waitingDialog;
    QLabel *waitingLabel;
    QPushButton *cancelButton;  // 新增取消按钮
    QTimer *waitingTimer;
    int waitingDots;
    QString currentTaskId;
    QEventLoop eventLoop;  // 用于同步等待
    bool requestFinished;  // 请求完成标志
    bool cancelRequested;  // 新增取消请求标志

    void showProgressDialog();
    void showWaitingDialog();
    void updateWaitingText();
    void hideWaitingDialog();
    void cancelGeneration();  // 新增取消生成函数
};

#endif // AICHANGEFACE_H

