import torch
import numpy as np
import os
import time
from collections import defaultdict
from tools import losses
import copy
from tools.mmd_loss import mmd_rbf
from torch.utils.tensorboard import SummaryWriter
import json
from federation.train_func.eval_model import eval_model
# torch.autograd.set_detect_anomaly(True)

def train_fedmine(config, model:torch.nn.Module, start_epoch, data_loader_train, data_loader_val, optimizer, log_dir,
                  scheduler, device, fed_round, clients_count, active_clients_len, global_feature, mmd, mmd_nobn):
    
    w1 = (2**config['alpha'] - 1) / 2**config['alpha']
    writer = SummaryWriter(log_dir)
    loss_epochs = []
    
    # 统计平均的本地特征
    # [batch_size, channel_num, height, width]
    client_feature = torch.zeros([8, 64, 2048]).to(device)
    # client_feature = torch.zeros([128, 4, 128]).to(device)
    
    if global_feature != None:
        global_feature = torch.flatten(global_feature, start_dim=1)
    
    t_start = time.time()
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        tb_train = defaultdict(list)
        model.train()
        
        client_feature_epoch = torch.zeros([8, 64, 2048]).to(device)
        # client_feature_epoch = torch.zeros([128, 4, 128]).to(device)
        frames_num = 0
        for index, (fid, x, _, _) in enumerate(data_loader_train):
            x = x.to(device)

            # Forward
            idx_valid, y, local_feature = model(x)
            # local_feature = local_feature.detach()
            
            # 计算输出的dwt和fft损失
            l_dwt, l_fft = losses.sparsity_loss(y)
            
            # 累加 batch 特征
            client_feature_epoch = client_feature_epoch + torch.sum(local_feature.data, dim=0) #  local_feature.sum(dim=0)
            frames_num = frames_num + local_feature.shape[0]

            # 残差图像(输出的干净图像-原始图像 = 噪声图像)
            residual = (y - x) * idx_valid
            # 计算残差图像的损失, 作为正则化项
            l_res = residual.abs().sum() / idx_valid.sum()

            # 输出的总损失
            l_sp = config['beta'] * l_dwt + (1 - config['beta']) * l_fft
            
            # 本地损失 = 输出图像的损失 + 残差图像的损失(正则化) , 用于更新个性化模块参数
            l_local = w1 * l_sp + (1 - w1) * l_res
            
            if mmd:
                # 用于更新共有参数的 全局损失(本地损失 + MMD(本地特征, 全局特征))
                local_feature = torch.flatten(local_feature, start_dim=1)
                if global_feature==None:
                    # global_feature = copy.deepcopy(local_feature)
                    # l_global = l_local.clone()
                    l_mmd = mmd_rbf(local_feature, local_feature.data)
                else:
                    l_mmd = mmd_rbf(local_feature, global_feature)
                l_global = l_local + 0.5 * config['mmd_mu'] * (l_mmd ** 2)
            else:
                l_global = l_local
            
            optimizer.zero_grad()
            if mmd_nobn and not config['personalized']:
                l_mmd = 0.5 * config['mmd_mu'] * (l_mmd ** 2)
                l_mmd.backward(retain_graph=True)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'bn' in name:
                            param.grad = None
                l_local.backward()
            elif mmd and (not mmd_nobn) and config['personalized']:
                l_mmd = 0.5 * config['mmd_mu'] * (l_mmd ** 2)
                l_mmd.backward(retain_graph=True)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'selfatt' in name:
                            param.grad = None
                l_local.backward()
            elif mmd_nobn and config['personalized']:
                l_mmd = 0.5 * config['mmd_mu'] * (l_mmd ** 2)
                l_mmd.backward(retain_graph=True)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'bn' in name or 'selfatt' in name:
                            param.grad = None
                l_local.backward()
            else:
                # 全局损失 反向传播 优化模型
                l_global.backward()
            optimizer.step()

            # elapsed time
            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join([
                f"rounds: [{(fed_round + 1):4d}/{config['num_rounds']:4d}]",
                f"clients: [{(clients_count + 1):4d}/{active_clients_len:4d}]",
                f"epochs: [{(epoch + 1):4d}/{(start_epoch + config['num_epochs']):4d}]",
                f'batches: [{(index + 1):4d}/{len(data_loader_train):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
                f'losses: {l_dwt.item():.6f} {l_fft.item():.6f} {l_res.item():.6f} ' ###{l_mmd.item():.6f}
            ]), end='\t\r')

            tb_train['loss/train/DWT'].append(l_dwt.item())
            tb_train['loss/train/FFT'].append(l_fft.item())
            tb_train['loss/train/Residual'].append(l_res.item())
            tb_train['loss/train/lsp'].append(l_sp.item())
            tb_train['loss/train/l_local'].append(l_local.item())
            if mmd:
                tb_train['loss/train/l_mmd'].append(l_mmd.item())
            tb_train['loss/train/l_global'].append(l_global.item())
            
        
        # 累加 epoch 特征
        client_feature = client_feature + (client_feature_epoch / frames_num)

        loss_epoch = {}
        for key, value in tb_train.items():
            loss_epoch[key] = np.nanmean(value)
            writer.add_scalar(key, np.nanmean(value), epoch + 1)
        loss_epochs.append(loss_epoch)

        # fn_ckpt = os.path.join(log_dir, f'epoch_{(epoch + 1):04d}.pth')
        # print(f'\nSaving {fn_ckpt:s} ...')
        # torch.save(model.state_dict(), fn_ckpt)

        # tb_val = defaultdict(list)
        # model.eval()
        # for index, (fid, x, _, _) in enumerate(data_loader_val):
        #     with torch.no_grad():
        #         x = x.to(device)

        #         idx_valid, y, _ = model.forward(x)
        #         l_dwt, l_fft = losses.sparsity_loss(y)

        #         residual = (y - x) * idx_valid
        #         l_res = residual.abs().sum() / idx_valid.sum()

        #         tb_val['loss/val/DWT'].append(l_dwt.item())
        #         tb_val['loss/val/FFT'].append(l_fft.item())
        #         tb_val['loss/val/Residual'].append(l_res.item())

        # for key, value in tb_val.items():
        #     writer.add_scalar(key, np.nanmean(value), epoch + 1)

        writer.flush()
        # 更新学习率
        scheduler.step()
    
    # 计算平均的本地特征, 最后维度是 [batch_size, channnel_num, height, width], batch_size维度是相同的特征, 主要是为了聚合的全局特征 与每次网络输出的本地特征维度一致 便于计算MMD loss
    client_feature = client_feature / config['num_epochs']
    client_feature = client_feature.unsqueeze(0)
    client_feature = client_feature.repeat(data_loader_train.batch_size, 1, 1, 1)

    writer.close()
    
    # 保存 client 在这一个round 在训练集上的loss
    loss_round = {}
    for key in loss_epochs[0].keys():
        loss_round[key] = np.mean([l_epoch[key] for l_epoch in loss_epochs])
    with open(os.path.join(log_dir, f'loss_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
        json.dump(loss_round, f, ensure_ascii=False, indent=4)
    
    # result_dict = eval_model(model, data_loader_train, config, device)
    # with open(os.path.join(log_dir, f'metrics_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
    #     json.dump(result_dict, f, ensure_ascii=False, indent=4)

    return client_feature, loss_round