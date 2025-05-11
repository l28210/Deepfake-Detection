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
from tools.eval_model import eval_model

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs. (default: 20)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size in each training step. (default: 32)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate. (default: 1e-3)')
parser.add_argument('--lr_decay', type=float, default=0.05,
                    help='Learning rate decay.')
parser.add_argument('--folder_data', type=str, default='/home/l/test_self/deepfake_detect/data',
                    help='folder of dataset')
parser.add_argument('--LARGE_KERNEL_CONV_IMPL', type=str, default='lib/RepLKNet-pytorch/cutlass/examples/19_large_depthwise_conv2d_torch_extension/',
                    help='hight efficient implementaion of conv')
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
# device = torch.device('cpu')



# tag取最后一段路径，无则使用时间作为tag
config['tag'] = config['tag'].split('/')[-1]
if not config['tag'].strip():
    config['tag'] = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# 保存参数文件，若已存在，则使用已存在的参数
log_dir = os.path.join(config['log_dir'], config['tag'])
os.makedirs(log_dir, exist_ok=True)
config_file = os.path.join(log_dir, 'config.json')
if os.path.exists(config_file):
    # Overwrite with saved config
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    # Save config to a JSON file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

# 学习率递减参数
if config['lr_decay'] < 0:
    config['lr_decay'] = 0.1**(1 / config['num_epochs'])


if config['dataset'] == 'archive':
    train_folder=os.path.join(config['folder_data'],'archive/train')
    valid_folder=os.path.join(config['folder_data'],'archive/valid')
    test_folder=os.path.join(config['folder_data'],'archive/test')

    ds_train=ForgeryDataset.example_gen(train_folder)
    ds_valid=ForgeryDataset.example_gen(valid_folder)
    ds_test=ForgeryDataset.example_gen(test_folder)

data_loader_train=DataLoader(
    ds_train,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['CPUs'],
    pin_memory=True,
)

data_loader_valid=DataLoader(
    ds_valid,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['CPUs'],
    pin_memory=True,
)

data_loader_test=DataLoader(
    ds_test,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['CPUs'],
    pin_memory=True,
)


os.environ['LARGE_KERNEL_CONV_IMPL'] = config['LARGE_KERNEL_CONV_IMPL']
model=create_RepLKNet31B(drop_path_rate=0.3, num_classes=2).to(device)

# checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
# if len(checkpoints):
#     ckpt = checkpoints[-1]
#     print(f'Loading the last checkpoint {ckpt:s}')
#     model.load_state_dict(torch.load(ckpt))

#     start_epoch = int(os.path.basename(ckpt).split('.')[0])
# else:
#     start_epoch = 0

start_epoch=0

writer = SummaryWriter(log_dir)

lr0 = config['lr'] * config['lr_decay']**start_epoch
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config['lr'],             
    weight_decay=config['lr_decay']         # 防止过拟合
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs'],             # 多少个epoch后cosine到最低点
    eta_min=1e-5                            # 最小学习率，训练到后期不会为0
)

t_start = time.time()
for epoch in range(start_epoch, config['num_epochs']):
    tb_train = defaultdict(list)
    model.train()
    for index, (images, labels) in enumerate(data_loader_train):
        # if index > 10:
        #     break
        images, labels = images.to(device), labels.to(device)
        # 前向
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # elapsed time
        mm, ss = divmod(time.time() - t_start, 60)
        hh, mm = divmod(mm, 60)

        print(' '.join([
            f"[{(epoch + 1):4d}/{config['num_epochs']:4d}]",
            f'[{(index + 1):4d}/{len(data_loader_train):4d}]',
            f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
            f'losses: {loss.item():.6f}'
        ]), end='\t\r')

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).float().sum()
        acc = correct / labels.size(0)

        tb_train['loss'].append(loss.item())
        tb_train['acc'].append(acc.item())

        # 此次batch的表现
        # if index % 20 == 0:
        print(f"[Train] Epoch {epoch}/{config['num_epochs']} Step {index}/{len(data_loader_train)} Loss: {loss.item():.4f} Acc: {acc.item():.4f}")

    # 每epoch一次valid
    tb_valid = defaultdict(list)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for index, (images, labels) in enumerate(data_loader_valid):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).float().sum()
            acc = correct / labels.size(0)

            tb_valid['loss'].append(loss.item())
            tb_valid['acc'].append(acc.item())

            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join(['[Validating]',
                f'[{(index + 1):4d}/{len(data_loader_valid):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
                f'losses: {loss.item():.6f}'
            ]), end='\t\r')

    avg_train_loss = np.mean(tb_train['loss'])
    avg_train_acc = np.mean(tb_train['acc'])
    avg_valid_loss = np.mean(tb_valid['loss'])
    avg_valid_acc = np.mean(tb_valid['acc'])

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_valid_loss:.4f} Val Acc: {avg_valid_acc:.4f}")

    # 写入 Tensorboard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
    writer.add_scalar('Acc/train', avg_train_acc, epoch)
    writer.add_scalar('Acc/valid', avg_valid_acc, epoch)

    scheduler.step()

    # 保存checkpoint
    torch.save(model.state_dict(), os.path.join(log_dir, f'{epoch}.pth'))

# 训练时间统计
t_end = time.time()
print(f"Finished Training! Total time: {(t_end - t_start)/60:.2f} mins.")

writer.close()