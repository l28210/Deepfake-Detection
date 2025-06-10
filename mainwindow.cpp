#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "AichangeFace.h"
#include "AIRecognition.h"
#include "MoreFunction.h"
#include <QStackedWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 连接按钮与对应页面的跳转
    connect(ui->AiChangeFace, &QPushButton::clicked, this, &MainWindow::openAiChangeFace);
    connect(ui->AIRecognition, &QPushButton::clicked, this, &MainWindow::openAIRecognition);
    connect(ui->MoreFunction, &QPushButton::clicked, this, &MainWindow::openMoreFunction);
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


