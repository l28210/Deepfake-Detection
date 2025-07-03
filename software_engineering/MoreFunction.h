#ifndef MOREFUNCTION_H
#define MOREFUNCTION_H

#include <QWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QFile>
#include <QFileDialog>
#include <QTimer>
#include <QEventLoop>
#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>

namespace Ui {
class MoreFunction;
}

class MoreFunction : public QWidget
{
    Q_OBJECT

public:
    explicit MoreFunction(QWidget *parent = nullptr);
    ~MoreFunction();

private slots:
    void returnToMain();
    void uploadModel();
    void clearFiles();
    void showHelp();
    void startGenerating();
    void downloadImage();
    void handleModelUploadResponse(QNetworkReply *reply);
    void handleHeatmapResponse(QNetworkReply *reply);
    void updateWaitingText();

private:
    Ui::MoreFunction *ui;
    QNetworkAccessManager *networkManager;
    QString modelFilePath;
    QString generatedImagePath;
    QString currentTaskId;
    QDialog *waitingDialog;
    QLabel *waitingLabel;
    QPushButton *cancelButton;
    QTimer *waitingTimer;
    int waitingDots;
    bool requestFinished;
    bool cancelRequested;
    QEventLoop eventLoop;

    void showWaitingDialog();
    void hideWaitingDialog();
    void cancelGeneration();
};

#endif // MOREFUNCTION_H
