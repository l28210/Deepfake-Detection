#include "MoreFunction.h"
#include "ui_MoreFunction.h"
#include "mainwindow.h"
#include <QMessageBox>
#include <QFile>
#include <QFileDialog>
#include <QJsonObject>
#include <QJsonDocument>
#include <QTimer>
#include <QEventLoop>
#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>
#include <QSettings>
#include <QDateTime>
#include <QThread>
#include <QDebug>
#include <QDesktopServices>  // 添加缺失的头文件
#include <QStandardPaths>

MoreFunction::MoreFunction(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MoreFunction),
    networkManager(new QNetworkAccessManager(this)),
    modelFilePath(""),
    generatedImagePath(""),
    currentTaskId(""),
    waitingDialog(nullptr),
    waitingLabel(nullptr),
    cancelButton(nullptr),
    waitingTimer(nullptr),
    waitingDots(0),
    requestFinished(false),
    cancelRequested(false)
{
    ui->setupUi(this);

    // 添加玻璃质感样式表
    QString glassStyle =
        "QPushButton {"
        "   background: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1,"
        "                               stop:0 rgba(0, 164, 255, 220),"
        "                               stop:1 rgba(222, 193, 255, 220));"
        "   border: 1px solid rgba(255, 255, 255, 100);"
        "   border-radius: 8px;"
        "   color: white;"
        "   padding: 8px;"
        "   font: bold 12pt \"Microsoft YaHei\";"
        "   min-width: 80px;"
        "   letter-spacing: 3px;"
        "}"
        "QPushButton:hover {"
        "   background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        "                               stop:0 rgba(120, 200, 255, 180),"
        "                               stop:1 rgba(70, 140, 220, 180));"
        "}"
        "QPushButton:pressed {"
        "   background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        "                               stop:0 rgba(80, 160, 235, 200),"
        "                               stop:1 rgba(40, 100, 180, 200));"
        "}";

    // 应用样式到所有按钮
    ui->ReturnMain->setStyleSheet(glassStyle);
    ui->UploadModel->setStyleSheet(glassStyle);
    ui->ClearFile->setStyleSheet(glassStyle);
    ui->Help->setStyleSheet(glassStyle);
    ui->StartGenerating->setStyleSheet(glassStyle);
    ui->DownloadImage->setStyleSheet(glassStyle);

    // 连接信号和槽
    connect(ui->ReturnMain, &QPushButton::clicked, this, &MoreFunction::returnToMain);
    connect(ui->UploadModel, &QPushButton::clicked, this, &MoreFunction::uploadModel);
    connect(ui->ClearFile, &QPushButton::clicked, this, &MoreFunction::clearFiles);
    connect(ui->Help, &QPushButton::clicked, this, &MoreFunction::showHelp);
    connect(ui->StartGenerating, &QPushButton::clicked, this, &MoreFunction::startGenerating);
    connect(ui->DownloadImage, &QPushButton::clicked, this, &MoreFunction::downloadImage);
    connect(networkManager, &QNetworkAccessManager::finished, this, &MoreFunction::handleModelUploadResponse);
    connect(networkManager, &QNetworkAccessManager::finished, this, &MoreFunction::handleHeatmapResponse);
}

MoreFunction::~MoreFunction()
{
    delete ui;
}

void MoreFunction::returnToMain()
{
    MainWindow *mainWindow = new MainWindow();
    mainWindow->show();
    this->hide();  // 隐藏当前界面
}

void MoreFunction::uploadModel()
{
    QString filePath = QFileDialog::getOpenFileName(this, "选择模型文件", "", "模型文件 (*.pth)");
    if (!filePath.isEmpty()) {
        modelFilePath = filePath;
        QFileInfo fileInfo(filePath);
        // 修正：使用正确的UI元素名称
        if (ui->ModelPathLabel) {  // 检查元素是否存在
            ui->ModelPathLabel->setText("已选择模型: " + fileInfo.fileName());
        }
        // QMessageBox::information(this, "成功", "模型文件已选择");
    }
}

void MoreFunction::clearFiles()
{
    modelFilePath = "";
    generatedImagePath = "";
    // 修正：使用正确的UI元素名称
    if (ui->ModelPathLabel) {  // 检查元素是否存在
        ui->ModelPathLabel->setText("未选择模型文件");
    }
    // QMessageBox::information(this, "已清除", "已清除所有选择的文件");
}

void MoreFunction::startGenerating()
{
    // 检查模型文件是否已选择
    if (modelFilePath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择模型文件！");
        return;
    }

    // 读取模型文件
    QFile file(modelFilePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "错误", "无法打开模型文件");
        return;
    }
    QByteArray modelData = file.readAll();
    file.close();

    // 获取服务器地址
    QSettings settings("MyCompany", "DeepFakeDetection");
    // QString serverAddress = settings.value("serverAddress", "http://127.0.0.1:5000").toString();
    // QString serverAddress = "http://26.27.68.168:5000";
    QString serverAddress = "http://127.0.0.1:5000";

    // 显示等待对话框
    showWaitingDialog();

    // 使用QTimer延迟执行网络请求，避免UI阻塞
    QTimer::singleShot(100, this, [this, serverAddress, modelData]() {
        // 创建JSON请求
        QJsonObject requestData;
        requestData["model_file"] = QString(modelData.toBase64());
        currentTaskId = QUuid::createUuid().toString();
        requestData["task_id"] = currentTaskId;

        // 重置状态
        requestFinished = false;
        cancelRequested = false;

        // 发送模型上传请求
        QNetworkRequest request(QUrl(serverAddress + "/api/upload_model"));
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        QNetworkReply *reply = networkManager->post(request, QJsonDocument(requestData).toJson());

        connect(reply, &QNetworkReply::finished, this, [this, reply, serverAddress]() {
            if (cancelRequested) {
                reply->deleteLater();
                return;
            }

            if (reply->error() != QNetworkReply::NoError) {
                QMessageBox::critical(this, "错误", "模型上传失败: " + reply->errorString());
            } else {
                QByteArray response = reply->readAll();
                QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
                QJsonObject jsonObject = jsonResponse.object();

                if (jsonObject["status"].toString() != "success") {
                    QMessageBox::warning(this, "错误", "模型上传失败: " + jsonObject["message"].toString());
                } else {
                    // 模型上传成功，直接请求生成热力图
                    // 不再需要用户选择图像

                    // 创建热力图生成请求
                    QJsonObject heatmapRequestData;
                    heatmapRequestData["task_id"] = currentTaskId;
                    // 不再需要image_data参数，后端会使用内置测试图像

                    QNetworkRequest heatmapRequest(QUrl(serverAddress + "/api/generate_heatmap"));
                    heatmapRequest.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
                    QNetworkReply *heatmapReply = networkManager->post(heatmapRequest, QJsonDocument(heatmapRequestData).toJson());

                    connect(heatmapReply, &QNetworkReply::finished, this, [this, heatmapReply, serverAddress]() {
                        if (cancelRequested) {
                            heatmapReply->deleteLater();
                            return;
                        }

                        if (heatmapReply->error() != QNetworkReply::NoError) {
                            QMessageBox::critical(this, "错误", "热力图生成失败: " + heatmapReply->errorString());
                        } else {
                            QByteArray heatmapResponse = heatmapReply->readAll();
                            QJsonDocument heatmapJsonResponse = QJsonDocument::fromJson(heatmapResponse);
                            QJsonObject heatmapJsonObject = heatmapJsonResponse.object();

                            if (heatmapJsonObject["status"].toString() != "success") {
                                QMessageBox::warning(this, "错误", "热力图生成失败: " + heatmapJsonObject["message"].toString());
                            } else {
                                // 解码生成的热力图
                                QString resultImageBase64 = heatmapJsonObject["result_image"].toString();
                                if (resultImageBase64.isEmpty()) {
                                    QMessageBox::warning(this, "错误", "生成的热力图数据为空");
                                } else {
                                    QByteArray imageData = QByteArray::fromBase64(resultImageBase64.toUtf8());
                                    QPixmap resultPixmap;
                                    if (!resultPixmap.loadFromData(imageData)) {
                                        QMessageBox::warning(this, "错误", "无法加载生成的热力图");
                                    } else {
                                        // 保存生成的热力图用于下载
                                        QDateTime timestamp = QDateTime::currentDateTime();
                                        generatedImagePath = "heatmap_" + timestamp.toString("yyyyMMddhhmmss") + ".png";
                                        if (!resultPixmap.save(generatedImagePath)) {
                                            QMessageBox::warning(this, "错误", "无法保存生成的热力图");
                                        }

                                        // 创建对话框显示结果
                                        QDialog *resultDialog = new QDialog(this);
                                        resultDialog->setWindowTitle("模型感受野热力图");
                                        resultDialog->setMinimumSize(800, 600);

                                        QLabel *imageLabel = new QLabel(resultDialog);
                                        imageLabel->setPixmap(resultPixmap.scaled(
                                            resultDialog->size(),
                                            Qt::KeepAspectRatio,
                                            Qt::SmoothTransformation
                                            ));
                                        imageLabel->setAlignment(Qt::AlignCenter);

                                        QVBoxLayout *layout = new QVBoxLayout(resultDialog);
                                        layout->addWidget(imageLabel);

                                        // 连接对话框关闭信号，确保资源释放
                                        connect(resultDialog, &QDialog::finished, [resultDialog]() {
                                            resultDialog->deleteLater();
                                        });

                                        resultDialog->show();

                                        // QMessageBox::information(this, "完成", "热力图生成成功！");
                                    }
                                }
                            }
                        }

                        heatmapReply->deleteLater();
                        requestFinished = true;
                        hideWaitingDialog();
                    });
                }
            }

            reply->deleteLater();
        });
    });
}

// 其余代码保持不变...

void MoreFunction::downloadImage()
{
    if (generatedImagePath.isEmpty()) {
        QMessageBox::warning(this, "警告", "热力图还未生成！");
        return;
    }

    QString savePath = QFileDialog::getSaveFileName(this, "保存热力图", "", "图像文件 (*.png *.jpg)");
    if (!savePath.isEmpty()) {
        QFile::copy(generatedImagePath, savePath);
        QMessageBox::information(this, "成功", "热力图已保存到：" + savePath);
    }
}

void MoreFunction::handleModelUploadResponse(QNetworkReply *reply)
{
    // 这个函数会被多次调用，需要根据reply的url判断是否是处理模型上传的响应
    // 实际处理在lambda函数中，这里不需要额外处理
}

void MoreFunction::handleHeatmapResponse(QNetworkReply *reply)
{
    // 这个函数会被多次调用，需要根据reply的url判断是否是处理热力图的响应
    // 实际处理在lambda函数中，这里不需要额外处理
}

void MoreFunction::showWaitingDialog()
{
    // 创建等待对话框
    waitingDialog = new QDialog(this, Qt::Dialog | Qt::FramelessWindowHint);
    waitingDialog->setMinimumSize(400, 180);
    waitingDialog->setWindowModality(Qt::ApplicationModal); // 阻塞其他窗口

    // 创建主布局
    QVBoxLayout *mainLayout = new QVBoxLayout(waitingDialog);
    mainLayout->setContentsMargins(30, 30, 30, 30);
    mainLayout->setSpacing(20);

    // 创建等待标签
    waitingLabel = new QLabel("正在上传模型", waitingDialog);
    waitingLabel->setAlignment(Qt::AlignCenter);
    waitingLabel->setStyleSheet("font: bold 14pt \"Microsoft YaHei\", \"SimHei\", \"Arial\", sans-serif; color: white; background: transparent;");
    waitingLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    waitingLabel->setWordWrap(true);

    // 创建取消按钮
    cancelButton = new QPushButton("取消", waitingDialog);
    cancelButton->setStyleSheet(
        "QPushButton {"
        "   background: rgba(255, 255, 255, 100);"
        "   border: 1px solid rgba(255, 255, 255, 150);"
        "   border-radius: 6px;"
        "   color: #333333;"
        "   padding: 6px 12px;"
        "   font: bold 12pt \"Microsoft YaHei\";"
        "}"
        "QPushButton:hover {"
        "   background: rgba(255, 255, 255, 150);"
        "}"
        "QPushButton:pressed {"
        "   background: rgba(255, 255, 255, 200);"
        "}"
        );
    cancelButton->setFixedWidth(100);
    cancelButton->setCursor(Qt::PointingHandCursor);

    // 连接取消按钮点击事件
    connect(cancelButton, &QPushButton::clicked, this, &MoreFunction::cancelGeneration);

    // 创建按钮布局，使按钮居中
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(cancelButton);
    buttonLayout->addStretch();

    // 添加到主布局
    mainLayout->addWidget(waitingLabel);
    mainLayout->addLayout(buttonLayout);

    // 设置样式表
    waitingDialog->setStyleSheet(
        "QDialog {"
        "   background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
        "                               stop:0 rgba(0, 164, 255, 220),"
        "                               stop:1 rgba(222, 193, 255, 220));"
        "   border-radius: 10px;"
        "}"
        );

    // 初始化等待点计数器
    waitingDots = 0;

    // 创建并启动定时器，每500毫秒更新一次等待文本
    waitingTimer = new QTimer(this);
    connect(waitingTimer, &QTimer::timeout, this, &MoreFunction::updateWaitingText);
    waitingTimer->start(500);

    // 显示对话框
    waitingDialog->show();
}

void MoreFunction::updateWaitingText()
{
    // 更新等待文本，添加点
    waitingDots = (waitingDots + 1) % 4;
    QString dots;
    for (int i = 0; i < waitingDots; i++) {
        dots += ".";
    }
    waitingLabel->setText("正在生成热力图" + dots);
}

void MoreFunction::hideWaitingDialog()
{
    // 停止定时器
    if (waitingTimer) {
        waitingTimer->stop();
        waitingTimer->deleteLater();
        waitingTimer = nullptr;
    }

    // 关闭对话框
    if (waitingDialog) {
        waitingDialog->accept();
        waitingDialog->deleteLater();
        waitingDialog = nullptr;
    }
}

void MoreFunction::cancelGeneration()
{
    // 设置取消标志
    cancelRequested = true;

    // 隐藏等待对话框
    hideWaitingDialog();

    // 显示取消提示
    QMessageBox::information(this, "已取消", "热力图生成已取消");
}

void MoreFunction::showHelp()
{
    QString resourcePath = ":/help/Receptive_Field.pdf";
    QString tempPath = QDir::toNativeSeparators(
        QStandardPaths::writableLocation(QStandardPaths::TempLocation) +
        "/webui_help_" +
        QDateTime::currentDateTime().toString("yyyyMMddhhmmss") +
        ".pdf"
        );

    // 检查资源文件
    QFile resFile(resourcePath);
    if (!resFile.exists() || !resFile.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "错误", "资源文件无效或损坏");
        return;
    }
    qDebug() << "Resource size:" << resFile.size() << "bytes";

    // 创建临时文件
    QFile tempFile(tempPath);
    if (tempFile.exists()) {
        if (!tempFile.remove()) {
            QMessageBox::critical(this, "错误", "无法清理旧临时文件");
            resFile.close();
            return;
        }
    }

    // 复制内容
    if (!tempFile.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "错误", "无法创建临时文件");
        resFile.close();
        return;
    }

    tempFile.write(resFile.readAll());
    tempFile.close();
    resFile.close();

    // 打开PDF
    // 替换后的跨平台代码
    if(!QDesktopServices::openUrl(QUrl::fromLocalFile(tempPath))) {
        QMessageBox::critical(this, "错误",
                              QString("无法打开PDF文件\n临时文件位置: %1").arg(tempPath));
    }
}
