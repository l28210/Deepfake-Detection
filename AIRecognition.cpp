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

AIRecognition::AIRecognition(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AIRecognition)
{
    ui->setupUi(this);

    //scene->setSceneRect(ui->showgraph->rect());

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

    QMessageBox::information(this, "成功", QString("已上传 %1 张图片").arg(fileNames.size()));
}

void AIRecognition::displayCurrentImage()
{
    scene->clear();  // 清空旧内容

    if (imageFileList.isEmpty()) {
        // 没有图片，显示提示文字
        QGraphicsTextItem *textItem = scene->addText("请上传图片");
        QFont font;
        font.setPointSize(14);  // 可调整字号
        textItem->setFont(font);

        // 获取scene的边界矩形，并居中显示文字
        QRectF textRect = textItem->boundingRect();
        QRectF sceneRect = ui->showgraph->sceneRect();

        // 计算居中位置
        qreal x = (sceneRect.width() - textRect.width()) / 2.0;
        qreal y = (sceneRect.height() - textRect.height()) / 2.0;
        textItem->setPos(x, y);
        ui->showgraph->setScene(scene);
        ui->ImageStatusLabel->setText("0/0");
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
    if (imageFolderPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先上传图片");
        return;
    }

    // 传输文件夹路径给后端（通过模拟命令行或API请求等）
    // 示例：通过QProcess调用后端程序（假设后端程序为识别程序）
    QString command = QString("python3 backend_recognition.py %1").arg(imageFolderPath);
    QProcess *process = new QProcess(this);
    process->start(command);

    if (!process->waitForStarted()) {
        QMessageBox::critical(this, "错误", "无法启动识别进程");
    } else {
        QMessageBox::information(this, "成功", "识别任务已开始");
    }
}

void AIRecognition::downloadReport()
{
    // 假设报告通过文件下载，打开文件选择对话框
    QString filePath = QFileDialog::getSaveFileName(this, "保存报告", "", "PDF 文件 (*.pdf)");
    if (filePath.isEmpty()) {
        return;
    }

    // 假设从后端获取报告并保存
    // 模拟下载报告
    QFile reportFile(filePath);
    if (reportFile.open(QIODevice::WriteOnly)) {
        // 假设报告内容存储在文件中（这里只是一个模拟）
        QTextStream out(&reportFile);
        out << "检测报告\n\n" << "结果：通过\n";
        reportFile.close();
        QMessageBox::information(this, "成功", "报告已下载");
    } else {
        QMessageBox::critical(this, "错误", "无法保存报告");
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
    QString resourcePath = ":/help/AIRecognition_help.pdf";
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
}

