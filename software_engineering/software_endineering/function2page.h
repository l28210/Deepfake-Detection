#ifndef FUNCTION2PAGE_H
#define FUNCTION2PAGE_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

class Function2Page : public QWidget
{
    Q_OBJECT
public:
    explicit Function2Page(QWidget *parent = nullptr);

signals:
    void returnToMainRequested();

private slots:
    void uploadBaseImage();
    void uploadRefImage();
    void startFaceSwap();
    void returnToMain();
    void clearImages();

private:
    QLabel *baseImageLabel;
    QLabel *refImageLabel;
    QPixmap baseImage;
    QPixmap refImage;
    QString baseImagePath;
    QString refImagePath;
};

#endif // FUNCTION2PAGE_H
