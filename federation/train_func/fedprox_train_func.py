import torch
import numpy as np
import os
import time
from collections import defaultdict
from tools import losses
from torch.utils.tensorboard import SummaryWriter
import json
import copy

def train_fedprox(config, model, start_epoch, data_loader_train, data_loader_val, optimizer, log_dir, scheduler, device, fed_round, clients_count, active_clients_len, _):
    w1 = (2**config['alpha'] - 1) / 2**config['alpha']
    writer = SummaryWriter(log_dir)
    loss_epochs = []
    
    global_model_params = [x for x in model.parameters()]
    
    t_start = time.time()
    for epoch in range(start_epoch, start_epoch+config['num_epochs']):
        tb_train = defaultdict(list)
        model.train()
        for index, (fid, x, _, _) in enumerate(data_loader_train):
            x = x.to(device)

            # Forward
            idx_valid, y, _ = model(x)
            l_dwt, l_fft = losses.sparsity_loss(y)

            residual = (y - x) * idx_valid
            l_res = residual.abs().sum() / idx_valid.sum()

            l_sp = config['beta'] * l_dwt + (1 - config['beta']) * l_fft
            loss = w1 * l_sp + (1 - w1) * l_res
            
            # FedProx 在FedAvg基础上加的正则化项
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model_params):
                proximal_term += (w - w_t).norm(2)
            loss_prox = loss + 0.5 * config['prox_mu'] * proximal_term

            # Backward and optimize
            optimizer.zero_grad()
            loss_prox.backward()
            optimizer.step()

            # elapsed time
            mm, ss = divmod(time.time() - t_start, 60)
            hh, mm = divmod(mm, 60)

            print(' '.join([
                f"rounds: [{(fed_round + 1):4d}/{config['num_rounds']:4d}]",
                f"clients: [{(clients_count + 1):4d}/{active_clients_len:4d}]",
                f"epochs: [{(epoch + 1):4d}/{(start_epoch+config['num_epochs']):4d}]",
                f'batches: [{(index + 1):4d}/{len(data_loader_train):4d}]',
                f'[{int(hh):02d}h{int(mm):02d}m{int(ss):02d}s]',
                f'losses: {l_dwt.item():.6f} {l_fft.item():.6f} {l_res.item():.6f}'
            ]), end='\t\r')

            tb_train['loss/train/DWT'].append(l_dwt.item())
            tb_train['loss/train/FFT'].append(l_fft.item())
            tb_train['loss/train/Residual'].append(l_res.item())
            tb_train['loss/train/lsp'].append(l_sp.item())
            tb_train['loss/train/loss'].append(loss.item())
            tb_train['loss/train/loss_prox'].append(loss_prox.item())

        loss_epoch = {}
        for key, value in tb_train.items():
            loss_epoch[key] = np.nanmean(value)
            writer.add_scalar(key, np.nanmean(value), epoch + 1)
        loss_epochs.append(loss_epoch)

        writer.flush()
        scheduler.step()

    writer.close()
    
    # 保存 client 在这一个round 在训练集上的loss
    loss_round = {}
    for key in loss_epochs[0].keys():
        loss_round[key] = np.mean([l_epoch[key] for l_epoch in loss_epochs])
    with open(os.path.join(log_dir, f'loss_round_{(fed_round + 1):04d}.json'), 'w', encoding='utf-8') as f:
        json.dump(loss_round, f, ensure_ascii=False, indent=4)
    
    return None, loss_round