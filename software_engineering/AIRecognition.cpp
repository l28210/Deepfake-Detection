#include "AIRecognition.h"
#include "ui_AIRecognition.h"
#include "mainwindow.h"
#include "customgraphicsview.h"
#include <QFileDialog>
#include <QDir>
#include <QProcess>
#include <QFile>
#include <QFileInfo>
#include <QDateTime>
#include <QCoreApplication>
#include <QMessageBox>
#include <QGraphicsView>
#include <QStandardPaths>
#include <QDesktopServices>
#include <QBuffer>  // 用于图像编码
#include <QJsonObject>  // 用于JSON处理
#include <QJsonDocument>
#include <QJsonArray>
#include <QDialog>  // 添加对话框头文件
#include <QLabel>   // 添加标签头文件
#include <QVBoxLayout>  // 添加布局头文件
#include <QPushButton>  // 添加按钮头文件
#include <QUrlQuery>
#include <QTimer>

AIRecognition::AIRecognition(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AIRecognition),
    scene(new QGraphicsScene(this))
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
        "   letter-spacing: 3px;"  // 新增字间距设置（单位：像素）"
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
    ui->UploadImage->setStyleSheet(glassStyle);
    ui->StartIdentify->setStyleSheet(glassStyle);
    ui->DownloadReport->setStyleSheet(glassStyle);
    ui->ClearImage->setStyleSheet(glassStyle);
    ui->Help->setStyleSheet(glassStyle);
    ui->ReturnMain->setStyleSheet(glassStyle);
    ui->ArrowLeft->setStyleSheet(glassStyle);
    ui->ArrowRight->setStyleSheet(glassStyle);

    //scene->setSceneRect(ui->showgraph->rect());
    networkManager = new QNetworkAccessManager(this);
    connect(networkManager, &QNetworkAccessManager::finished,
            this, &AIRecognition::handleDetectionResponse);

    // 连接按钮与槽函数
    connect(ui->UploadImage, &QPushButton::clicked, this, &AIRecognition::uploadImage);
    connect(ui->StartIdentify, &QPushButton::clicked, this, &AIRecognition::startIdentify);
    connect(ui->DownloadReport, &QPushButton::clicked, this, &AIRecognition::downloadReport);
    connect(ui->ClearImage, &QPushButton::clicked, this, &AIRecognition::clearImage);
    connect(ui->Help, &QPushButton::clicked, this, &AIRecognition::showHelp);
    connect(ui->ReturnMain, &QPushButton::clicked, this, &AIRecognition::returnToMain);

    scene = new QGraphicsScene(this);
    ui->showgraph->setScene(scene);
    ui->showgraph->setRenderHint(QPainter::Antialiasing);

    connect(ui->ArrowLeft, &QPushButton::clicked, this, &AIRecognition::showPreviousImage);
    connect(ui->ArrowRight, &QPushButton::clicked, this, &AIRecognition::showNextImage);
    connect(ui->showgraph, &CustomGraphicsView::showlargergraph, this, &AIRecognition::enlargeImage);

    // 初始化时显示"请上传图片"提示
    displayCurrentImage();
}

AIRecognition::~AIRecognition()
{
    delete ui;
}

void AIRecognition::uploadImage()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(this, "选择图片", QDir::homePath(), "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileNames.isEmpty()) {
        QMessageBox::warning(this, "警告", "未选择图片");
        return;
    }

    QString baseDir = QCoreApplication::applicationDirPath() + "/uploaded_images";
    QDir().mkpath(baseDir);
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMddHHmmss");
    imageFolderPath = baseDir + "/" + timestamp;
    QDir().mkpath(imageFolderPath);

    imageFileList.clear();  // 清空旧图
    for (const QString &file : fileNames) {
        QFileInfo fileInfo(file);
        QString destPath = imageFolderPath + "/" + fileInfo.fileName();
        QFile::copy(file, destPath);
        imageFileList.append(destPath);
    }

    currentIndex = 0;
    displayCurrentImage();

    // QMessageBox::information(this, "成功", QString("已上传 %1 张图片").arg(fileNames.size()));
}

void AIRecognition::displayCurrentImage()
{
    scene->clear();  // 清空旧内容

    if (imageFileList.isEmpty()) {
        // 没有图片，显示提示文字
        // QGraphicsTextItem *textItem = scene->addText("请上传图片");
        // QFont font;
        // font.setPointSize(14);  // 可调整字号

        // textItem->setFont(font);
        QGraphicsTextItem *textItem = scene->addText(""); // 先创建空文本项
        textItem->setHtml("<span style='color: white;font: bold 20pt \"Microsoft YaHei\"; '>请上传图片</span>");

        // 获取scene的边界矩形，并居中显示文字
        QRectF textRect = textItem->boundingRect();
        QRectF sceneRect = ui->showgraph->sceneRect();

        // 计算居中位置
        qreal x = (sceneRect.width() - textRect.width()) / 2.0;
        qreal y = (sceneRect.height() - textRect.height()) / 2.0;
        textItem->setPos(x, y);
        ui->ImageStatusLabel->setText("0/0");
        ui->showgraph->setScene(scene);
        return;
    }

    // 有图片时显示当前图像
    if (currentIndex < 0 || currentIndex >= imageFileList.size()) return;

    QPixmap pix(imageFileList[currentIndex]);
    if (pix.isNull()) return;

    scene->addPixmap(pix.scaled(ui->showgraph->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    scene->update();

    ui->showgraph->setScene(scene);

    // 显示页码
    ui->ImageStatusLabel->setText(QString("%1/%2").arg(currentIndex + 1).arg(imageFileList.size()));
}

void AIRecognition::showPreviousImage()
{
    if (imageFileList.isEmpty()) return;
    currentIndex = (currentIndex - 1 + imageFileList.size()) % imageFileList.size();
    displayCurrentImage();
}

void AIRecognition::showNextImage()
{
    if (imageFileList.isEmpty()) return;
    currentIndex = (currentIndex + 1) % imageFileList.size();
    displayCurrentImage();
}

void AIRecognition::enlargeImage()
{
    if (imageFileList.isEmpty()) return;

    QDialog *dialog = new QDialog(this);
    dialog->setWindowTitle("图片放大");
    dialog->resize(800, 600);

    QLabel *label = new QLabel(dialog);
    label->setAlignment(Qt::AlignCenter);
    QPixmap pix(imageFileList[currentIndex]);
    label->setPixmap(pix.scaled(dialog->size(), Qt::KeepAspectRatio));
    label->setGeometry(0, 0, dialog->width(), dialog->height());

    dialog->exec();
}

void AIRecognition::startIdentify()
{
    if (imageFileList.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先上传图片");
        return;
    }

    // 显示等待对话框
    showWaitingDialog();

    // 获取服务器地址
    QSettings settings("MyCompany", "DeepFakeDetection");
    // QString serverAddress = settings.value("serverAddress", "http://127.0.0.1:5000").toString();
    // QString serverAddress = "http://26.27.68.168:5000";
    QString serverAddress = "http://127.0.0.1:5000";
    settings.sync();  // 确保马上写入

    // 直接读取原始文件
    QString imgPath = imageFileList[currentIndex];
    QFile file(imgPath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "错误", "无法打开图片文件");
        hideWaitingDialog(); // 隐藏等待对话框
        return;
    }
    QByteArray imageData = file.readAll();
    file.close();

    QString taskId = QUuid::createUuid().toString();

    qDebug() << "[AIRecognition] 准备发送检测请求，服务器地址：" << serverAddress
             << "，图像路径：" << imgPath
             << "，taskId：" << taskId;

    QJsonObject requestData;
    requestData["image_data"] = QString(imageData.toBase64());
    requestData["task_id"] = taskId;

    QNetworkRequest request(QUrl(serverAddress + "/api/detect"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    // 发送请求并保存回复对象
    currentReply = networkManager->post(request, QJsonDocument(requestData).toJson());

    qDebug() << "[AIRecognition] 网络请求已发送，正在等待服务器响应...";

    // 实时上传进度
    connect(currentReply, &QNetworkReply::uploadProgress,
            this, [](qint64 bytesSent, qint64 bytesTotal) {
                qDebug() << "[AIRecognition] 上传进度："
                         << bytesSent << "/" << bytesTotal;
            });

    // 连接信号槽处理响应，使用Lambda表达式以便访问cancelRequested标志
    connect(currentReply, &QNetworkReply::finished, this, [this]() {
        if (cancelRequested) {
            // 如果用户已请求取消，忽略响应
            currentReply->deleteLater();
            currentReply = nullptr;
            return;
        }

        handleDetectionResponse(currentReply);
        currentReply = nullptr;
    });
}

void AIRecognition::handleDetectionResponse(QNetworkReply *reply)
{
    if (reply->error() != QNetworkReply::NoError) {
        hideWaitingDialog(); // 隐藏等待对话框
        QMessageBox::critical(this, "错误", "网络请求失败: " + reply->errorString());
        reply->deleteLater();
        return;
    }

    QByteArray response = reply->readAll();
    reply->deleteLater();

    QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
    QJsonObject jsonObject = jsonResponse.object();

    if (jsonObject["status"].toString() == "success") {
        if (jsonObject.contains("is_fake") &&
            jsonObject.contains("fake_confidence") &&
            jsonObject.contains("real_confidence")) {

            bool isFake = jsonObject["is_fake"].toBool();
            double fakeConf = jsonObject["fake_confidence"].toDouble();
            double realConf = jsonObject["real_confidence"].toDouble();

            showDetectionResult(isFake, fakeConf, realConf);
        } else {
            hideWaitingDialog(); // 隐藏等待对话框
            QMessageBox::warning(this, "错误", "检测结果不完整，请重试");
        }
    } else {
        hideWaitingDialog(); // 隐藏等待对话框
    }
}

// 实现 showDetectionResult 函数
void AIRecognition::showDetectionResult(bool isFake, double fakeConf, double realConf)
{
    // 显示检测结果
    QString result = QString("检测结果: %1\n伪造评分值: %2\n真实评分值: %3")
                         .arg(isFake ? "伪造" : "真实")
                         .arg(fakeConf, 0, 'f', 4)
                         .arg(realConf, 0, 'f', 4);

    // 创建结果对话框
    QDialog *resultDialog = new QDialog(this);
    resultDialog->setWindowTitle("AI识别结果");
    resultDialog->setMinimumSize(400, 200);

    // 创建标签显示结果
    QLabel *resultLabel = new QLabel(resultDialog);
    resultLabel->setText(result);
    resultLabel->setAlignment(Qt::AlignCenter);
    resultLabel->setWordWrap(true);

    // 创建确定按钮
    QPushButton *okButton = new QPushButton("确定", resultDialog);
    okButton->setStyleSheet("QPushButton {"
                            "   background: qlineargradient(x1:0.5, y1:0, x2:0.5, y2:1,"
                            "                               stop:0 rgba(0, 164, 255, 220),"
                            "                               stop:1 rgba(222, 193, 255, 220));"
                            "   border: 1px solid rgba(255, 255, 255, 100);"
                            "   border-radius: 8px;"
                            "   color: white;"
                            "   padding: 8px;"
                            "   font: bold 12pt \"Microsoft YaHei\";"
                            "   min-width: 80px;"
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
                            "}");
    connect(okButton, &QPushButton::clicked, resultDialog, &QDialog::accept);

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(resultDialog);
    layout->addWidget(resultLabel);
    layout->addWidget(okButton, 0, Qt::AlignCenter);

    // 显示对话框
    resultDialog->exec();

    // 清理资源
    delete resultDialog;
}

void AIRecognition::showWaitingDialog()
{
    // 创建等待对话框
    waitingDialog = new QDialog(this, Qt::Dialog | Qt::FramelessWindowHint);
    waitingDialog->setMinimumSize(400, 180); // 增加高度以容纳按钮
    waitingDialog->setWindowModality(Qt::ApplicationModal); // 阻塞其他窗口

    // 创建主布局
    QVBoxLayout *mainLayout = new QVBoxLayout(waitingDialog);
    mainLayout->setContentsMargins(30, 30, 30, 30);
    mainLayout->setSpacing(20);

    // 创建等待标签
    waitingLabel = new QLabel("正在识别中", waitingDialog);
    waitingLabel->setAlignment(Qt::AlignCenter);
    waitingLabel->setStyleSheet("font: bold 14pt \"Microsoft YaHei\"; color: white; background: transparent;");
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
    connect(cancelButton, &QPushButton::clicked, this, &AIRecognition::cancelDetection);

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

    // 初始化等待点计数器和状态
    waitingDots = 0;
    requestFinished = false;
    cancelRequested = false;

    // 创建并启动定时器，每500毫秒更新一次等待文本
    waitingTimer = new QTimer(this);
    connect(waitingTimer, &QTimer::timeout, this, &AIRecognition::updateWaitingText);
    waitingTimer->start(500);

    // 显示对话框
    waitingDialog->show();
}

void AIRecognition::cancelDetection()
{
    // 设置取消标志
    cancelRequested = true;

    // 中止网络请求
    if (currentReply && currentReply->isRunning()) {
        currentReply->abort();
        currentReply->deleteLater();
        currentReply = nullptr;
    }

    // 隐藏等待对话框
    hideWaitingDialog();

    // 显示取消提示
    QMessageBox::information(this, "已取消", "识别过程已取消");
}

void AIRecognition::downloadReport()
{
    if (detectionResults.isEmpty()) {
        QMessageBox::warning(this, "警告", "没有可下载的检测结果");
        return;
    }

    // 创建报告内容
    QString reportContent = "AI识别检测报告\n\n";
    reportContent += "检测时间: " + QDateTime::currentDateTime().toString() + "\n\n";

    for (int i = 0; i < imageFileList.size(); ++i) {
        if (i < detectionResults.size()) {
            QJsonObject result = detectionResults.values().at(i);
            reportContent += QString("图片 %1: %2\n伪造概率: %3%\n真实概率: %4%\n\n")
                                 .arg(i + 1)
                                 .arg(result["is_fake"].toBool() ? "伪造" : "真实")
                                 .arg(result["fake_probability"].toDouble() * 100, 0, 'f', 2)
                                 .arg(result["real_probability"].toDouble() * 100, 0, 'f', 2);
        }
    }

    // 保存报告
    QString savePath = QFileDialog::getSaveFileName(this, "保存报告", "", "文本文件 (*.txt)");
    if (!savePath.isEmpty()) {
        QFile reportFile(savePath);
        if (reportFile.open(QIODevice::WriteOnly)) {
            reportFile.write(reportContent.toUtf8());
            reportFile.close();
            QMessageBox::information(this, "成功", "报告已下载");
        } else {
            QMessageBox::critical(this, "错误", "无法保存报告");
        }
    }
}

void AIRecognition::updateWaitingText()
{
    // 更新等待文本，添加点
    waitingDots = (waitingDots + 1) % 4;
    QString dots;
    for (int i = 0; i < waitingDots; i++) {
        dots += ".";
    }
    waitingLabel->setText("正在识别中" + dots);
}

void AIRecognition::hideWaitingDialog()
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

void AIRecognition::clearImage()
{
    // imageFolderPath.clear();
    // QMessageBox::information(this, "成功", "已清空上传的图片");
    // 清空所有图片相关数据
    imageFolderPath.clear();
    imageFileList.clear();
    currentIndex = 0;

    // 更新界面显示
    displayCurrentImage();

    QMessageBox::information(this, "成功", "已清空上传的图片");
}

void AIRecognition::showHelp()
{
    QString resourcePath = ":/help/AI_Recognition.pdf";
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

void AIRecognition::returnToMain()
{
    MainWindow *mainWindow = new MainWindow();
    mainWindow->show();
    this->hide();  // 隐藏当前界面
    this->raise();  // 确保在最上层
}

