#ifndef AICHANGEFACE_H
#define AICHANGEFACE_H

#include <QWidget>

namespace Ui {
class AiChangeFace;
}

class AiChangeFace : public QWidget
{
    Q_OBJECT

public:
    explicit AiChangeFace(QWidget *parent = nullptr);
    ~AiChangeFace();

private slots:
    void returnToMain();

private:
    Ui::AiChangeFace *ui;
};

#endif // AICHANGEFACE_H

