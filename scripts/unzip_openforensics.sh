#!/bin/bash
set -e

cd ~/test_self/deepfake_detect/data/OpenForensics

mkdir -p OpenForensics_processed/{train,val,test-dev,test-challenge}

# 解压训练
for f in Train_part_*.zip; do unzip -q "$f" -d OpenForensics_processed/train; done
# 解压验证
unzip -q Val.zip -d OpenForensics_processed/val
# 解压测试-dev
for f in Test-Dev_part_*.zip; do unzip -q "$f" -d OpenForensics_processed/test-dev; done
# 解压测试-challenge
for f in Test-Challenge_part_*.zip; do unzip -q "$f" -d OpenForensics_processed/test-challenge; done

# 拷贝标注
cp Train_poly.json OpenForensics_processed/train/
cp Val_poly.json OpenForensics_processed/val/
cp Test-Dev_poly.json OpenForensics_processed/test-dev/
cp Test-Challenge_poly.json OpenForensics_processed/test-challenge/
