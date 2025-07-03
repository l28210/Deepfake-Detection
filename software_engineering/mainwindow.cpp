#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "AichangeFace.h"
#include "AIRecognition.h"
#include "MoreFunction.h"
#include <QStackedWidget>
#include <QInputDialog>  // 添加头文件
#include <QMessageBox>  // 用于提示框
#include <QLabel>       // 用于 scrollTextLabel

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    settings("MyCompany", "DeepFakeDetection")
{
    ui->setupUi(this);
    this->setWindowIcon(QIcon("://DeepFakeDetection.ico"));

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
        "   font: bold 16pt \"Microsoft YaHei\";"
        "   min-width: 80px;"
        "   letter-spacing: 3px;"  // 字间距设置"
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
    ui->AiChangeFace->setStyleSheet(glassStyle);
    ui->AIRecognition->setStyleSheet(glassStyle);
    ui->MoreFunction->setStyleSheet(glassStyle);
    ui->contactus->setStyleSheet(glassStyle);

    // 连接按钮与对应页面的跳转
    connect(ui->AiChangeFace, &QPushButton::clicked, this, &MainWindow::openAiChangeFace);
    connect(ui->AIRecognition, &QPushButton::clicked, this, &MainWindow::openAIRecognition);
    connect(ui->MoreFunction, &QPushButton::clicked, this, &MainWindow::openMoreFunction);

    // connect(ui->contactus, &QPushButton::clicked, [this]() {
    //     bool ok;
    //     QString serverAddress = QInputDialog::getText(this, tr("联系我们"),
    //                                                   tr("邮箱地址"), QLineEdit::Normal,
    //                                                   settings.value("MailAddress", "2981237274@qqcom").toString(), &ok);

    //     if (ok && !serverAddress.isEmpty()) {
    //         settings.setValue("serverAddress", serverAddress);
    //         QMessageBox::information(this, tr("成功"), tr("谢谢使用"));
    //     }
    // });

    connect(ui->contactus, &QPushButton::clicked, [this]() {
        QInputDialog dialog(this);
        dialog.setWindowTitle("联系我们");
        dialog.setLabelText("邮箱地址：");
        dialog.setTextValue(settings.value("MailAddress", "petrichoka.xiao@foxmail.com").toString());

        // 设置输入框、提示文字和按钮的样式
        dialog.setStyleSheet(R"(
        /* 整个 QInputDialog 的背景 */
        QDialog,QInputDialog {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 rgba(255, 255, 255, 220),
                                        stop:1 rgba(220, 235, 255, 200));
            border-radius: 12px;
            border: 2px solid #66ccff;
        }

        /* 提示文字 QLabel */
        QLabel {
            color: #333;
            font: 13pt "Microsoft YaHei";
            background: transparent;
        }

        /* 邮箱输入框 QLineEdit */
        QLineEdit {
            background: rgba(255, 255, 255, 200);
            border: 2px solid #a0cfff;
            border-radius: 10px;
            padding: 6px;
            font: 12pt "Microsoft YaHei";
            color: #003366;
        }

        /* OK / Cancel 按钮 QPushButton */
        QPushButton {
            background: rgba(102, 204, 255, 200);
            border: none;
            border-radius: 8px;
            padding: 6px 20px;
            font: bold 11pt "Microsoft YaHei";
            color: white;
            min-width: 80px; /* 新增属性：设置最小宽度 */
        }

        QPushButton:hover {
            background: rgba(51, 170, 255, 220);
        }

        QPushButton:pressed {
            background: rgba(30, 144, 255, 230);
        }
    )");

        if (dialog.exec() == QDialog::Accepted) {
            QString serverAddress = dialog.textValue();
            if (!serverAddress.isEmpty()) {
                settings.setValue("serverAddress", serverAddress);

                // 美化成功提示框
                QMessageBox msgBox(QMessageBox::Information, "成功", "谢谢使用", QMessageBox::Ok, this);
                msgBox.setStyleSheet(R"(
                QMessageBox {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                stop:0 rgba(240, 250, 255, 230),
                                                stop:1 rgba(210, 235, 255, 220));
                    border-radius: 12px;
                    border: 2px solid #66ccff;
                    font: 11pt "Microsoft YaHei";
                }
                QLabel {
                    color: #333;
                    background: transparent;
                }
                QPushButton {
                    background: #66ccff;
                    border-radius: 6px;
                    padding: 6px 16px;
                    font: bold 10pt "Microsoft YaHei";
                    color: white;
                    min-width: 80px; /* 新增属性：设置最小宽度 */
                }
                QPushButton:hover {
                    background: #33bbff;
                }
            )");
                msgBox.exec();
            }
        }
    });

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::openAiChangeFace()
{
    AiChangeFace *page = new AiChangeFace();
    page->show();
    this->hide();  // 隐藏当前主界面
}

void MainWindow::openAIRecognition()
{
    AIRecognition *page = new AIRecognition();
    page->show();
    this->hide();  // 隐藏当前主界面
}

void MainWindow::openMoreFunction()
{
    MoreFunction *page = new MoreFunction();
    page->show();
    this->hide();  // 隐藏当前主界面
}


QString MainWindow::getServerAddress() const
{
    return settings.value("serverAddress", "http://26.27.68.168:5000").toString();
}
