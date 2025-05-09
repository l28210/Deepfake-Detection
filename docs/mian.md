# Deepfake Detection
> 初步思路:对原始RGB图像通过`ConvNeXt`或`ReplkNet`(超大核CNN)达到图像分类的效果,后续计划进一部训练达到图像分割的效果(或神经网络可解释性),为了增强模型准确率,考虑通过傅里叶变换等转化为频域图像,在原有模型基础上增加模块或额外使用一种模型增强效果
## 不同CNN架构优势
模型 | 主要特点 | 适用任务
---|---|---
ResNet | 残差连接，解决梯度消失问题 | 通用视觉任务
EfficientNet | 复合缩放 + NAS 设计，计算高效 | 轻量级分类
ConvNeXt | 纯 CNN 模仿 Transformer，大核卷积 | 分类、检测、分割
[RepLKNet](https://arxiv.org/pdf/2203.06717) | 超大核 CNN (31×31)，适合高分辨率任务 | 目标检测、分割
CoAtNet | CNN + Transformer 结合，提高计算效率 | 分类、检测
EdgeNeXt | 轻量级 CNN + Transformer，适合移动端 | 移动端 AI

## 开发环境
- Ubuntu 20.04
- python 3.9.0
- torch 2.6.0+cu118

## 数据来源
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)


## model
***`replknet`***
> [replknet](./reference/ReplKNet.pdf)    
[RepLKNet作者解读：超大卷积核，大到31x31，越大越暴力，涨点又高效！（CVPR 2022）](https://zhuanlan.zhihu.com/p/481445076)    
[深入浅出理解深度可分离卷积（Depthwise Separable Convolution）](https://blog.csdn.net/m0_37605642/article/details/134174749)

`replknet`主要通过大卷积核提升有效感受野同时结合深度可分离卷积优化计算性能  
[详细代码解析](./replknet.md)

## load data
项目 | Dataset | DataLoader
---|---|---
职责 | 定义数据的读取方式：每次返回一张图像和标签（__getitem__） | 负责批量加载数据，可以打乱、并行读取、批量返回
行为 | Dataset[i] → 返回第 i 个样本 | for batch in DataLoader: → 每次返回一批（batch）
数据单位 | 单个数据点（图像、标签） | 一批数据点，默认 batch 是 list/stack 后的 Tensor

## optimizer & scheduler
> [Adam和AdamW](https://zhuanlan.zhihu.com/p/643452086)  
scheduler使用余弦退火学习率