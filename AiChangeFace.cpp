#include "AichangeFace.h"
#include "ui_AichangeFace.h"
#include "mainwindow.h"

AiChangeFace::AiChangeFace(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::AiChangeFace)
{
    ui->setupUi(this);
    connect(ui->ReturnMain, &QPushButton::clicked, this, &AiChangeFace::returnToMain);
}

AiChangeFace::~AiChangeFace()
{
    delete ui;
}

void AiChangeFace::returnToMain()
{
    MainWindow *mainWindow = new MainWindow();
    mainWindow->show();
    this->hide();  // 隐藏当前界面
}


