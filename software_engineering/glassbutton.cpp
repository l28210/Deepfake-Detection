#include "glassbutton.h"

GlassButton::GlassButton(QWidget *parent)
    : QPushButton(parent)
{
    setAttribute(Qt::WA_Hover);
    setCursor(Qt::PointingHandCursor);

    // 初始化点击动画
    m_clickAnimation = new QPropertyAnimation(this, "opacityEffect");
    m_clickAnimation->setDuration(200);
    m_clickAnimation->setEasingCurve(QEasingCurve::OutQuad);

    QGraphicsDropShadowEffect *shadow = new QGraphicsDropShadowEffect(this);
    shadow->setBlurRadius(10);
    shadow->setColor(QColor(0, 100, 255, 80));
    shadow->setOffset(0, 3);
    this->setGraphicsEffect(shadow);
}


void GlassButton::mousePressEvent(QMouseEvent *e)
{
    QPushButton::mousePressEvent(e);
    m_clickAnimation->stop();
    m_clickAnimation->setStartValue(1.0);
    m_clickAnimation->setEndValue(0.7);
    m_clickAnimation->start();
}

void GlassButton::mouseReleaseEvent(QMouseEvent *e)
{
    QPushButton::mouseReleaseEvent(e);
    m_clickAnimation->stop();
    m_clickAnimation->setStartValue(opacityEffect());
    m_clickAnimation->setEndValue(1.0);
    m_clickAnimation->start();
}

qreal GlassButton::opacityEffect() const
{
    return m_opacityEffect;
}

void GlassButton::setOpacityEffect(qreal opacity)
{
    if (qFuzzyCompare(m_opacityEffect, opacity))
        return;

    m_opacityEffect = opacity;
    update();
    emit opacityEffectChanged(m_opacityEffect);
}

GlassButton::GlassButton(const QString &text, QWidget *parent)
    : QPushButton(text, parent)
{
    setAttribute(Qt::WA_Hover);
    setCursor(Qt::PointingHandCursor);
}

void GlassButton::setGlassColor(const QColor &topColor, const QColor &bottomColor)
{
    m_topColor = topColor;
    m_bottomColor = bottomColor;
    update();
}

void GlassButton::setHighlightColor(const QColor &color)
{
    m_highlightColor = color;
    update();
}

void GlassButton::setBorderColor(const QColor &color)
{
    m_borderColor = color;
    update();
}

void GlassButton::setCornerRadius(int radius)
{
    m_cornerRadius = radius;
    update();
}

void GlassButton::setTextColor(const QColor &color)
{
    m_textColor = color;
    update();
}

void GlassButton::animateClick()
{
    if (!m_clickAnimation) return;

    m_clickAnimation->stop();
    m_clickAnimation->setStartValue(1.0);
    m_clickAnimation->setEndValue(0.7);
    m_clickAnimation->start();
}

void GlassButton::paintEvent(QPaintEvent *e)
{
    Q_UNUSED(e)

    QPainter painter(this);
    painter.setOpacity(m_opacityEffect);
    painter.setRenderHint(QPainter::Antialiasing);

    QRect rect = this->rect();

    // 绘制基础玻璃底色
    QLinearGradient gradient(0, 0, 0, height());
    gradient.setColorAt(0, m_topColor);
    gradient.setColorAt(1, m_bottomColor);

    painter.setBrush(gradient);
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(rect, m_cornerRadius, m_cornerRadius);

    // 绘制高光效果
    if (m_hovered) {
        QLinearGradient highlight(0, 0, 0, height()/3);
        highlight.setColorAt(0, m_highlightColor);
        highlight.setColorAt(1, Qt::transparent);

        painter.setBrush(highlight);
        painter.drawRoundedRect(rect, m_cornerRadius, m_cornerRadius);
    }

    // 绘制边框
    QPen pen(m_borderColor);
    pen.setWidth(1);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), m_cornerRadius, m_cornerRadius);

    // 绘制文本
    painter.setPen(m_textColor);
    QFont font = this->font();
    font.setBold(true);
    painter.setFont(font);
    painter.drawText(rect, Qt::AlignCenter, text());
}

void GlassButton::enterEvent(QEnterEvent *event)
{
    Q_UNUSED(event)
    m_hovered = true;
    update();
}

void GlassButton::leaveEvent(QEvent *event)
{
    Q_UNUSED(event)
    m_hovered = false;
    update();
}
