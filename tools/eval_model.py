import sys
sys.path.append('/home/l/test_self/deepfake_detect/')

import torch
import argparse
import os
import json
import glob
import time
import numpy as np


from collections import defaultdict
from datetime import datetime
from tools.utils import set_seed
from tools.datasets import ForgeryDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.replknet import create_RepLKNet31B
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def eval_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    runtimes = []
    t_start=time.time()

    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            start_time = time.time()
            logits = model(images)
            runtime = time.time() - start_time
            runtimes.append(runtime / images.size(0))  # 平均每张图时间

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join(['[Testing]',
                f'[{(index + 1):4d}/{len(loader):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]']), 
                end='\t\r')

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Runtime(ms)": np.median(runtimes) * 1000
    }

    return metrics_dict

if __name__ == '__main__':
    sys.path.append('/home/l/test_self/deepfake_detect/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs. (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size in each training step. (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. (default: 1e-3)')
    parser.add_argument('--lr_decay', type=float, default=0.05,
                        help='Learning rate decay.')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataset', type=str, default='archive', choices=['archive'])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--CPUs', type=int, default=16,
                        help='Number of CPU used for train.(default: 16)')
    config = vars(parser.parse_args())

    # 固定随机种子
    set_seed(config['seed'])

    # 选择运行的GPU
    device = torch.device('cuda')

    os.environ['LARGE_KERNEL_CONV_IMPL'] = 'lib/RepLKNet-pytorch/cutlass/examples/19_large_depthwise_conv2d_torch_extension/'
    model=create_RepLKNet31B(drop_path_rate=0.3, num_classes=2).to(device)

    ds_train=ForgeryDataset.example_gen('/home/l/test_self/deepfake_detect/data/archive/train')

    data_loader_train=DataLoader(
        ds_train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['CPUs'],
        pin_memory=True,
    )

    test_metrics = eval_model(model, data_loader_train, device)
    print(f"Acc: {test_metrics['Accuracy']:.4f} "
        f"Precision: {test_metrics['Precision']:.4f} "
        f"Recall: {test_metrics['Recall']:.4f} "
        f"F1: {test_metrics['F1']:.4f}")