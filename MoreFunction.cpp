#include "MoreFunction.h"
#include "ui_MoreFunction.h"
#include "mainwindow.h"

MoreFunction::MoreFunction(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MoreFunction)
{
    ui->setupUi(this);
    connect(ui->ReturnMain, &QPushButton::clicked, this, &MoreFunction::returnToMain);
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

