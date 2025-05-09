#include "function1page.h"

Function1Page::Function1Page(QWidget *parent) : QWidget(parent)
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // 标题
    QLabel *title = new QLabel("识别AI", this);
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet("font-size: 24px; font-weight: bold;");

    // 图片显示区域
    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setFixedSize(400, 400);
    imageLabel->setStyleSheet("border: 2px dashed #aaa;");

    // 按钮区域
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    QPushButton *uploadBtn = new QPushButton("上传图片", this);
    QPushButton *removeBtn = new QPushButton("删除图片", this);
    QPushButton *detectBtn = new QPushButton("开始识别", this);
    QPushButton *returnBtn = new QPushButton("返回主菜单", this);

    buttonLayout->addWidget(uploadBtn);
    buttonLayout->addWidget(removeBtn);
    buttonLayout->addWidget(detectBtn);
    buttonLayout->addWidget(returnBtn);

    // 组装布局
    mainLayout->addWidget(title);
    mainLayout->addWidget(imageLabel);
    mainLayout->addLayout(buttonLayout);

    // 连接信号槽
    connect(uploadBtn, &QPushButton::clicked, this, &Function1Page::uploadImage);
    connect(removeBtn, &QPushButton::clicked, this, &Function1Page::removeImage);
    connect(detectBtn, &QPushButton::clicked, this, &Function1Page::startDetection);
    connect(returnBtn, &QPushButton::clicked, this, &Function1Page::returnToMain);


    // 拖放支持
    setAcceptDrops(true);
}


void Function1Page::uploadImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "选择图片", "", "图片文件 (*.jpg *.png)");
    if(!fileName.isEmpty()) {
        QFileInfo fileInfo(fileName);
        if(fileInfo.size() > 1024*1024) {
            QMessageBox::warning(this, "错误", "图片大小不能超过1MB");
            return;
        }

        currentImage.load(fileName);
        imagePath = fileName;
        imageLabel->setPixmap(currentImage.scaled(imageLabel->size(), Qt::KeepAspectRatio));
    }
}

void Function1Page::removeImage() {
    currentImage = QPixmap();
    imagePath.clear();
    imageLabel->clear();
    imageLabel->setText("请上传图片");
}

void Function1Page::startDetection() {
    if(imagePath.isEmpty()) {
        QMessageBox::warning(this, "错误", "请先上传图片");
        return;
    }

    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "确认", "确定要开始识别吗?", QMessageBox::Yes|QMessageBox::No);

    if(reply == QMessageBox::Yes) {
        // 调用AI识别功能
        QMessageBox::information(this, "提示", "识别功能将在后端实现");
    }
}

void Function1Page::returnToMain() {
    // 检查是否有未保存数据
    if (!imagePath.isEmpty()) {
        QMessageBox::StandardButton reply = QMessageBox::warning(
            this,
            "未完成操作",
            "当前已上传图片未处理，确定要放弃吗？",
            QMessageBox::Discard | QMessageBox::Cancel
            );

        if (reply != QMessageBox::Discard) {
            return;  // 用户取消，返回
        }
    }

    // 清理资源
    removeImage();

    // 触发信号
    emit returnToMainRequested();

    // 可选：添加返回动画
    // QPropertyAnimation *anim = new QPropertyAnimation(this, "windowOpacity");
    // anim->setDuration(300);
    // anim->setStartValue(1.0);
    // anim->setEndValue(0.0);
    // anim->start(QAbstractAnimation::DeleteWhenStopped);
    //增加直接拖拽图片功能
    //返回主菜单函数修改
    //图片放置框位置调整，title位置调整
}
