#ifndef FUNCTION1PAGE_H
#define FUNCTION1PAGE_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDialog>

class Function1Page : public QWidget
{
    Q_OBJECT
public:
    explicit Function1Page(QWidget *parent = nullptr);

signals:
    void returnToMainRequested();  // 新增信号

private slots:
    void uploadImage();
    void removeImage();
    void startDetection();
    void returnToMain();

private:
    QLabel *imageLabel;
    QPixmap currentImage;
    QString imagePath;
};

#endif // FUNCTION1PAGE_H
