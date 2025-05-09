import sys
sys.path.append('/home/l/test_self/deepfake_detect')

import cv2
import numpy as np
import torch
from model.replknet import *

# 读取图片
image = cv2.imread("/home/l/test_self/deepfake_detect/data/archive/real_vs_fake/real-vs-fake/test/fake/0COHPC8N1I.jpg")  # 请替换成你的图片路径
# OpenCV 读取的是 BGR 格式，需要转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取 RGB 矩阵
rgb_matrix = np.array(image_rgb)
print(rgb_matrix.shape)  # (height, width, 3)

model=create_RepLKNet31B(num_classes=2)

x=torch.Tensor(rgb_matrix)

print(x.shape)

# 转换为 float32 类型并归一化
rgb_matrix = rgb_matrix.astype(np.float32) / 255.0

# 转换为 tensor，并调整维度顺序
x = torch.from_numpy(rgb_matrix).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)

print(x.shape)  # 现在应该是 torch.Size([1, 3, 256, 256])

res=model(x)

print(res)

label=torch.tensor([1])

loss_crossloss=torch.nn.CrossEntropyLoss()

loss_sigmoid=torch.nn.BCEWithLogitsLoss()

# 假设标签是类别 0（真实），注意是 int 类型，不需要 one-hot
target = torch.tensor([1])  # [batch_size]，对应于类别索引

loss=loss_crossloss(res,target)

print(loss)