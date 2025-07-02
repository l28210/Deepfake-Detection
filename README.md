# Federated Deepfake Detection

## Introduction

Deepfake detection is becoming increasingly critical due to the rapid growth of AI-generated media. Traditional deep learning methods often rely on centralized data collection, raising concerns about data privacy and scalability. 

This project proposes a federated learning framework based on **RepLKNet**, enabling collaborative Deepfake detection without sharing raw data. By leveraging large-kernel convolution and client-side training, the system aims to preserve privacy while maintaining high accuracy.

---

## Dataset

- [FaceForensics++](https://github.com/ltnghia/openforensics)
- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

---

## Directory Structure

```bash
Deepfake-Detection
├── data                        # Dataset storage
├── environment.yml             # Conda environment config
├── eval_central.py             # Centralized evaluation script
├── federation                  # Federated learning methods
├── img                         # Visualization or sample images
├── lib                         # RepLKNet and dependencies
├── LICENSE
├── model                       # Training models and saved checkpoints
├── README.md
├── requirements.txt
├── scripts                     # Training script wrappers
├── software_engineering        # UI design components
├── temp.npy                    # Temporary result
├── test                        # Test utilities
├── tools                       # Helper tools
├── train_central.py            # Centralized training script
├── train_fedavg.py             # FedAvg training script
├── train_fedDyn.py             # FedDyn training script
├── train_fednorm.py            # FedNorm training script
├── train_fedprox.py            # FedProx training script
└── visualize                   # ERF computation and visualization
```

## Getting Started
```bash
git clone --recurse-submodules https://github.com/l28210/Deepfake-Detection.git
conda env create -f environment.yml
```
environment : Ubuntu 20.04 + CUDA 11.8 + nvcc 11.8 + cudnn 9.1.0 + python 3.9.0 + pytorch 2.6.0 + gcc 9.4.0 + NVIDIA driver 535.183.01

More details can be found in [requirement](./requirements.txt) and [environment](./environment.yml)

- Efficient Large-Kernel Convolution with PyTorch

To use [efficient large-kernel convolution with PyTorch](https://github.com/MegEngine/cutlass/tree/master/examples/19_large_depthwise_conv2d_torch_extension), , compile the custom extension:
```bash
# compile
cd lib/RepLKNet-pytorch/
unzip  cutlass.zip 
cd cutlass/examples/19_large_depthwise_conv2d_torch_extension
chmod +x setup.py
./setup.py install --user
```

- central trainnig
```bash
# trainning and evaling
./run_central.sh
```

