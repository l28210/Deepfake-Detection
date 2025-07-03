#ifndef GLASSBUTTON_H
#define GLASSBUTTON_H

#include <QPushButton>
#include <QPainter>
#include <QLinearGradient>
#include <QEvent>
#include <QEnterEvent>
#include <QPen>
#include <QPropertyAnimation>
#include <QGraphicsDropShadowEffect>

class GlassButton : public QPushButton
{
    Q_OBJECT
    Q_PROPERTY(qreal opacityEffect READ opacityEffect WRITE setOpacityEffect NOTIFY opacityEffectChanged) // 添加属性声明

public:
    explicit GlassButton(QWidget *parent = nullptr);
    explicit GlassButton(const QString &text, QWidget *parent = nullptr);

    // 设置玻璃效果颜色
    void setGlassColor(const QColor &topColor, const QColor &bottomColor);
    // 设置高光颜色
    void setHighlightColor(const QColor &color);
    // 设置边框颜色
    void setBorderColor(const QColor &color);
    // 设置圆角半径
    void setCornerRadius(int radius);
    // 设置文本颜色
    void setTextColor(const QColor &color);

    // 添加属性访问函数
    qreal opacityEffect() const;
    void setOpacityEffect(qreal opacity);

signals:
    void opacityEffectChanged(qreal opacity); // 添加信号

protected:
    void paintEvent(QPaintEvent *e) override;
    void enterEvent(QEnterEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void mousePressEvent(QMouseEvent *e) override;
    void mouseReleaseEvent(QMouseEvent *e) override;

private slots:
    void animateClick();

private:
    QPropertyAnimation *m_clickAnimation;
    qreal m_opacityEffect = 1.0;

    QColor m_topColor = QColor(100, 180, 255, 150);
    QColor m_bottomColor = QColor(50, 120, 200, 150);
    QColor m_highlightColor = QColor(255, 255, 255, 80);
    QColor m_borderColor = QColor(200, 230, 255, 100);
    QColor m_textColor = Qt::white;
    int m_cornerRadius = 8;
    bool m_hovered = false;
};

#endif // GLASSBUTTON_H
