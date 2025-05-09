#include "function2page.h"
#include <QFileDialog>
#include <QMessageBox>

Function2Page::Function2Page(QWidget *parent) : QWidget(parent)
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // 标题
    QLabel *title = new QLabel("AI换脸", this);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet("font-size: 24px; font-weight: bold;");

    // 图片区域
    QHBoxLayout *imageLayout = new QHBoxLayout();

    QVBoxLayout *baseLayout = new QVBoxLayout();
    QLabel *baseTitle = new QLabel("底图", this);
    baseTitle->setAlignment(Qt::AlignCenter);
    baseImageLabel = new QLabel(this);
    baseImageLabel->setAlignment(Qt::AlignCenter);
    baseImageLabel->setFixedSize(300, 300);
    baseImageLabel->setStyleSheet("border: 2px dashed #aaa;");
    QPushButton *uploadBaseBtn = new QPushButton("上传底图", this);

    baseLayout->addWidget(baseTitle);
    baseLayout->addWidget(baseImageLabel);
    baseLayout->addWidget(uploadBaseBtn);

    QVBoxLayout *refLayout = new QVBoxLayout();
    QLabel *refTitle = new QLabel("参考", this);
    refTitle->setAlignment(Qt::AlignCenter);
    refImageLabel = new QLabel(this);
    refImageLabel->setAlignment(Qt::AlignCenter);
    refImageLabel->setFixedSize(300, 300);
    refImageLabel->setStyleSheet("border: 2px dashed #aaa;");
    QPushButton *uploadRefBtn = new QPushButton("上传参考图", this);

    refLayout->addWidget(refTitle);
    refLayout->addWidget(refImageLabel);
    refLayout->addWidget(uploadRefBtn);

    imageLayout->addLayout(baseLayout);
    imageLayout->addLayout(refLayout);

    // 操作按钮
    QHBoxLayout *actionLayout = new QHBoxLayout();
    QPushButton *clearBtn = new QPushButton("清空图片", this);
    QPushButton *swapBtn = new QPushButton("开始换脸", this);
    QPushButton *returnBtn = new QPushButton("返回主菜单", this);

    actionLayout->addWidget(clearBtn);
    actionLayout->addWidget(swapBtn);
    actionLayout->addWidget(returnBtn);

    // 组装布局
    mainLayout->addWidget(title);
    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(actionLayout);

    // 连接信号槽
    connect(uploadBaseBtn, &QPushButton::clicked, this, &Function2Page::uploadBaseImage);
    connect(uploadRefBtn, &QPushButton::clicked, this, &Function2Page::uploadRefImage);
    connect(swapBtn, &QPushButton::clicked, this, &Function2Page::startFaceSwap);
    connect(returnBtn, &QPushButton::clicked, this, &Function2Page::returnToMain);
    connect(clearBtn, &QPushButton::clicked, this, &Function2Page::clearImages);

    // 拖放支持
    setAcceptDrops(true);
}


void Function2Page::uploadBaseImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "选择底图", "", "图片文件 (*.jpg *.png)");
    if(!fileName.isEmpty()) {
        // 类似function1的实现
    }
}

void Function2Page::uploadRefImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "选择参考图", "", "图片文件 (*.jpg *.png)");
    if(!fileName.isEmpty()) {
        // 类似function1的实现
    }
}

void Function2Page::startFaceSwap() {
    if(baseImagePath.isEmpty() || refImagePath.isEmpty()) {
        QMessageBox::warning(this, "错误", "请上传底图和参考图");
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "确认", "确定要开始换脸吗?", QMessageBox::Yes|QMessageBox::No);

    if(reply == QMessageBox::Yes) {
        // 调用换脸功能
        QMessageBox::information(this, "提示", "换脸功能将在后端实现");
    }
}

void Function2Page::clearImages() {
    baseImage = QPixmap();
    refImage = QPixmap();
    baseImagePath.clear();
    refImagePath.clear();
    baseImageLabel->clear();
    refImageLabel->clear();
}

void Function2Page::returnToMain() {
    // 或者直接调用父窗口的方法
    emit returnToMainRequested();
}
