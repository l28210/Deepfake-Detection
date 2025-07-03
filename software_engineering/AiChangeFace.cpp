#include "AichangeFace.h"
#include "ui_AichangeFace.h"
#include "mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QDesktopServices>
#include <QUrl>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QBuffer>  // 用于图像编码
#include <QJsonObject>  // 用于JSON处理
#include <QJsonDocument>
#include <QDialog>  // 添加对话框头文件
#include <QLabel>   // 添加标签头文件
#include <QVBoxLayout>  // 添加布局头文件
#include <QTimer>
#include <QStandardPaths>

AiChangeFace::AiChangeFace(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AiChangeFace),
    mainScene(nullptr),
    referenceScene(nullptr),
    generatedScene(nullptr),
    mainPixmapItem(nullptr),
    referencePixmapItem(nullptr),
    generatedPixmapItem(nullptr),
    currentReply(nullptr)  // 添加初始化
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
    ui->ReturnMain->setStyleSheet(glassStyle);
    ui->UploadMainImage->setStyleSheet(glassStyle);
    ui->UploadReferenceImage->setStyleSheet(glassStyle);
    ui->StartGenerating->setStyleSheet(glassStyle);
    ui->ClearImage->setStyleSheet(glassStyle);
    ui->DownloadImage->setStyleSheet(glassStyle);
    ui->Help->setStyleSheet(glassStyle);

    // 初始化网络管理器
    networkManager = new QNetworkAccessManager(this);
    connect(networkManager, &QNetworkAccessManager::finished,
            this, &AiChangeFace::handleFaceSwapResponse);

    // 初始化场景
    mainScene = new QGraphicsScene(this);
    referenceScene = new QGraphicsScene(this);
    generatedScene = new QGraphicsScene(this);

    // 设置场景到视图
    ui->graphicsView->setScene(mainScene);
    ui->graphicsView_2->setScene(referenceScene);

    // 初始化状态
    mainImageUploaded = false;
    referenceImageUploaded = false;
    imageGenerated = false;

    connect(ui->ReturnMain, &QPushButton::clicked, this, &AiChangeFace::returnToMain);
    connect(ui->UploadMainImage, &QPushButton::clicked, this, &AiChangeFace::uploadMainImage);
    connect(ui->UploadReferenceImage, &QPushButton::clicked, this, &AiChangeFace::uploadReferenceImage);
    connect(ui->StartGenerating, &QPushButton::clicked, this, &AiChangeFace::startGenerating);
    connect(ui->ClearImage, &QPushButton::clicked, this, &AiChangeFace::clearImages);
    connect(ui->DownloadImage, &QPushButton::clicked, this, &AiChangeFace::downloadImage);
    connect(ui->Help, &QPushButton::clicked, this, &AiChangeFace::showHelp);
}

AiChangeFace::~AiChangeFace()
{
    if (currentReply) {
        currentReply->deleteLater();
        currentReply = nullptr;
    }
    delete ui;
    delete mainScene;
    delete referenceScene;
    delete generatedScene;
}

void AiChangeFace::returnToMain()
{
    MainWindow *mainWindow = new MainWindow();
    mainWindow->show();
    this->hide();  // 隐藏当前界面
}

void AiChangeFace::uploadMainImage()
{
    QString filePath = QFileDialog::getOpenFileName(this, "选择主图", "", "图片文件 (*.png *.jpg *.jpeg)");
    if (!filePath.isEmpty()) {
        QPixmap pixmap(filePath);
        if (!pixmap.isNull()) {
            // 清除现有图片项
            if (mainPixmapItem) {
                mainScene->removeItem(mainPixmapItem);
                delete mainPixmapItem;
            }

            // 创建新的图片项并添加到场景
            mainPixmapItem = new QGraphicsPixmapItem(pixmap);
            mainScene->addItem(mainPixmapItem);

            // 调整视图以显示整个图片
            ui->graphicsView->fitInView(mainPixmapItem, Qt::KeepAspectRatio);

            // 保存图片数据
            QBuffer buffer(&sourceImageData);
            buffer.open(QIODevice::WriteOnly);
            pixmap.save(&buffer, "JPG");

            mainImageUploaded = true;
        } else {
            QMessageBox::warning(this, "错误", "无法加载图片文件");
        }
    }
}

void AiChangeFace::uploadReferenceImage()
{
    QString filePath = QFileDialog::getOpenFileName(this, "选择参考图", "", "图片文件 (*.png *.jpg *.jpeg)");
    if (!filePath.isEmpty()) {
        QPixmap pixmap(filePath);
        if (!pixmap.isNull()) {
            // 清除现有图片项
            if (referencePixmapItem) {
                referenceScene->removeItem(referencePixmapItem);
                delete referencePixmapItem;
            }

            // 创建新的图片项并添加到场景
            referencePixmapItem = new QGraphicsPixmapItem(pixmap);
            referenceScene->addItem(referencePixmapItem);

            // 调整视图以显示整个图片
            ui->graphicsView_2->fitInView(referencePixmapItem, Qt::KeepAspectRatio);

            // 保存图片数据
            QBuffer buffer(&targetImageData);
            buffer.open(QIODevice::WriteOnly);
            pixmap.save(&buffer, "JPG");

            referenceImageUploaded = true;
        } else {
            QMessageBox::warning(this, "错误", "无法加载图片文件");
        }
    }
}

void AiChangeFace::startGenerating()
{
    // 检查图片上传状态
    if (!mainImageUploaded && !referenceImageUploaded) {
        QMessageBox::warning(this, "警告", "请先上传主图和参考图！");
        return;
    } else if (!mainImageUploaded) {
        QMessageBox::warning(this, "警告", "请先上传主图！");
        return;
    } else if (!referenceImageUploaded) {
        QMessageBox::warning(this, "警告", "请先上传参考图！");
        return;
    }

    // 获取服务器地址
    QSettings settings("MyCompany", "DeepFakeDetection");
    // QString serverAddress = settings.value("serverAddress", "http://127.0.0.1:5000").toString();
    // QString serverAddress = "http://26.27.68.168:5000";
    QString serverAddress = "http://127.0.0.1:5000";

    // 创建JSON请求
    QJsonObject requestData;
    requestData["source_image"] = QString(sourceImageData.toBase64());
    requestData["target_image"] = QString(targetImageData.toBase64());
    currentTaskId = QUuid::createUuid().toString();
    requestData["task_id"] = currentTaskId;

    QNetworkRequest request(QUrl(serverAddress + "/api/face_swap"));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    // 重置状态
    requestFinished = false;
    cancelRequested = false;

    // 显示进度对话框
    showProgressDialog();

    // 发送请求
    QNetworkReply *reply = networkManager->post(request, QJsonDocument(requestData).toJson());

    // 保存回复对象的指针，以便取消时使用
    currentReply = reply;

    // 连接信号槽处理响应
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (cancelRequested) {
            // 如果用户已请求取消，忽略响应
            reply->deleteLater();
            return;
        }

        if (reply->error() != QNetworkReply::NoError) {
            // QMessageBox::critical(this, "错误", "网络请求失败: " + reply->errorString());
        } else {
            QByteArray response = reply->readAll();

            QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
            QJsonObject jsonObject = jsonResponse.object();

            if (jsonObject["status"].toString() != "success") {
                // QMessageBox::warning(this, "错误", "图片生成失败: " + jsonObject["message"].toString());
            } else {
                // 解码生成的图片
                QString resultImageBase64 = jsonObject["result_image"].toString();
                if (resultImageBase64.isEmpty()) {
                    QMessageBox::warning(this, "错误", "生成的图片数据为空");
                    hideWaitingDialog();
                    return;
                }

                QByteArray imageData = QByteArray::fromBase64(resultImageBase64.toUtf8());
                QPixmap resultPixmap;
                if (!resultPixmap.loadFromData(imageData)) {
                    QMessageBox::warning(this, "错误", "无法加载生成的图片");
                    hideWaitingDialog();
                    return;
                }

                // 保存生成的图片用于下载
                QDateTime timestamp = QDateTime::currentDateTime();
                generatedImagePath = "generated_image_" + timestamp.toString("yyyyMMddhhmmss") + ".jpg";
                if (!resultPixmap.save(generatedImagePath)) {
                    QMessageBox::warning(this, "错误", "无法保存生成的图片");
                }

                // 创建对话框显示结果图片
                QDialog *resultDialog = new QDialog(this);
                resultDialog->setWindowTitle("AI换脸结果");
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
                connect(resultDialog, &QDialog::finished, [this, resultDialog]() {
                    resultDialog->deleteLater();
                });

                resultDialog->show();

                imageGenerated = true;
                QMessageBox::information(this, "完成", "图片生成成功！");
            }
        }

        reply->deleteLater();
        requestFinished = true;
        hideWaitingDialog(); // 隐藏进度对话框
    });
}

void AiChangeFace::handleFaceSwapResponse(QNetworkReply *reply)
{
    if (reply->error() != QNetworkReply::NoError) {
        // QMessageBox::critical(this, "错误", "网络请求失败: " + reply->errorString());
        reply->deleteLater();
        return;
    }

    QByteArray response = reply->readAll();
    reply->deleteLater();

    QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
    QJsonObject jsonObject = jsonResponse.object();

    if (jsonObject["status"].toString() != "success") {
        // QMessageBox::warning(this, "错误", "图片生成失败: " + jsonObject["message"].toString());
        return;
    }

    // 解码生成的图片
    QByteArray imageData = QByteArray::fromBase64(jsonObject["result_image"].toString().toUtf8());
    QPixmap resultPixmap;
    resultPixmap.loadFromData(imageData);

    // 保存生成的图片用于下载
    generatedImagePath = "generated_image.jpg";
    resultPixmap.save(generatedImagePath);

    // 创建对话框显示图片
    QDialog *imageDialog = new QDialog(this);
    imageDialog->setWindowTitle("生成结果");
    imageDialog->setMinimumSize(800, 600);

    // 创建标签显示图片
    QLabel *imageLabel = new QLabel(imageDialog);
    imageLabel->setPixmap(resultPixmap.scaled(800, 600, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    imageLabel->setAlignment(Qt::AlignCenter);

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(imageDialog);
    layout->addWidget(imageLabel);

    // 显示对话框
    imageDialog->exec();

    // 清理资源
    delete imageDialog;

    imageGenerated = true;
    // QMessageBox::information(this, "完成", "图片生成成功！");
}

void AiChangeFace::showProgressDialog()
{
    // 创建进度对话框
    waitingDialog = new QDialog(this, Qt::Dialog | Qt::FramelessWindowHint);
    waitingDialog->setMinimumSize(400, 180); // 增加高度以容纳按钮
    waitingDialog->setWindowModality(Qt::ApplicationModal); // 阻塞其他窗口

    // 创建主布局
    QVBoxLayout *mainLayout = new QVBoxLayout(waitingDialog);
    mainLayout->setContentsMargins(30, 30, 30, 30);
    mainLayout->setSpacing(20);

    // 创建等待标签
    waitingLabel = new QLabel("正在处理图片", waitingDialog);
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
    connect(cancelButton, &QPushButton::clicked, this, &AiChangeFace::cancelGeneration);

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
    connect(waitingTimer, &QTimer::timeout, this, &AiChangeFace::updateWaitingText);
    waitingTimer->start(500);

    // 显示对话框
    waitingDialog->show();
}

void AiChangeFace::cancelGeneration()
{
    // 设置取消标志
    cancelRequested = true;

    // 中止网络请求
    if (currentReply && currentReply->isRunning()) {
        currentReply->abort();
        currentReply->deleteLater();
    }

    // 隐藏等待对话框
    hideWaitingDialog();

    // 显示取消提示
    // QMessageBox::information(this, "已取消", "图片生成已取消");
}

void AiChangeFace::showWaitingDialog()
{
    // 创建等待对话框
    waitingDialog = new QDialog(this, Qt::Dialog | Qt::FramelessWindowHint);
    waitingDialog->setMinimumSize(300, 150); // 增加高度
    waitingDialog->setWindowModality(Qt::ApplicationModal); // 阻塞其他窗口

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(waitingDialog);
    layout->setContentsMargins(30, 30, 30, 30); // 设置边距
    layout->setAlignment(Qt::AlignCenter);      // 设置居中对齐

    // 创建等待标签
    waitingLabel = new QLabel("正在生成图片", waitingDialog);
    waitingLabel->setAlignment(Qt::AlignCenter);
    waitingLabel->setStyleSheet("font: bold 14pt \"Microsoft YaHei\", \"SimHei\", \"Arial\", sans-serif; color: white; background: transparent;");
    waitingLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // 允许标签扩展
    waitingLabel->setWordWrap(true); // 允许文本换行

    // 添加到布局
    layout->addWidget(waitingLabel);

    // 设置样式表
    waitingDialog->setStyleSheet(
        "QDialog {"
        "   background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
        "                               stop:0 rgba(0, 164, 255, 220),"
        "                               stop:1 rgba(222, 193, 255, 220));"
        "   border-radius: 10px;"
        "   padding: 20px;" // 添加内边距
        "}"
        );

    // 初始化等待点计数器
    waitingDots = 0;

    // 创建并启动定时器，每500毫秒更新一次等待文本
    waitingTimer = new QTimer(this);
    connect(waitingTimer, &QTimer::timeout, this, &AiChangeFace::updateWaitingText);
    waitingTimer->start(500);

    // 显示对话框
    waitingDialog->show();
}

void AiChangeFace::updateWaitingText()
{
    // 更新等待文本，添加点
    waitingDots = (waitingDots + 1) % 4;
    QString dots;
    for (int i = 0; i < waitingDots; i++) {
        dots += ".";
    }
    waitingLabel->setText("正在生成图片" + dots);
}

void AiChangeFace::hideWaitingDialog()
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

void AiChangeFace::clearImages()
{
    // 清除主图
    if (mainPixmapItem) {
        mainScene->removeItem(mainPixmapItem);
        delete mainPixmapItem;
        mainPixmapItem = nullptr;
    }
    mainImageUploaded = false;

    // 清除参考图
    if (referencePixmapItem) {
        referenceScene->removeItem(referencePixmapItem);
        delete referencePixmapItem;
        referencePixmapItem = nullptr;
    }
    referenceImageUploaded = false;
}

void AiChangeFace::downloadImage()
{
    if (!imageGenerated) {
        QMessageBox::warning(this, "警告", "图片还未生成！");
        return;
    }

    QString savePath = QFileDialog::getSaveFileName(this, "保存图片", "", "图片文件 (*.png *.jpg)");
    if (!savePath.isEmpty()) {
        QFile::copy(generatedImagePath, savePath);
        QMessageBox::information(this, "成功", "图片已保存到：" + savePath);
    }
}

void AiChangeFace::showHelp()
{
    QString resourcePath = ":/help/AIFace_Swap.pdf";
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
