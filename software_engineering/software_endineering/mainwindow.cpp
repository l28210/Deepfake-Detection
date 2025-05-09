#include "mainwindow.h"
#include "function1page.h"
#include "function2page.h"
#include <QPushButton>
#include <QVBoxLayout>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle("AI功能工具箱");
    setFixedSize(500, 600);

    stackedWidget = new QStackedWidget(this);

    // 主菜单
    QWidget *mainMenu = new QWidget();
    QVBoxLayout *menuLayout = new QVBoxLayout();

    QPushButton *btnFunc1 = new QPushButton("AI图片检测");
    QPushButton *btnFunc2 = new QPushButton("AI换脸");
    QPushButton *btnExit = new QPushButton("退出");

    menuLayout->addWidget(btnFunc1);
    menuLayout->addWidget(btnFunc2);
    menuLayout->addStretch();
    menuLayout->addWidget(btnExit);
    mainMenu->setLayout(menuLayout);

    //创建子页面
    Function1Page *func1Page = new Function1Page(this);
    Function2Page *func2Page = new Function2Page(this);

    // 添加页面
    stackedWidget->addWidget(func1Page);
    stackedWidget->addWidget(func2Page);
    stackedWidget->addWidget(mainMenu);
    stackedWidget->setCurrentIndex(2);

    // 连接信号槽
    connect(btnFunc1, &QPushButton::clicked, this, [=](){ stackedWidget->setCurrentIndex(0); });
    connect(btnFunc2, &QPushButton::clicked, this, [=](){ stackedWidget->setCurrentIndex(1); });
    connect(btnExit, &QPushButton::clicked, this, &QMainWindow::close);


    connect(func1Page, &Function1Page::returnToMainRequested,this, [=](){stackedWidget->setCurrentIndex(2);});
    connect(func2Page, &Function2Page::returnToMainRequested,this, [=](){stackedWidget->setCurrentIndex(2);});

    setCentralWidget(stackedWidget);
}





