#!/bin/bash

# 定义基础命令
base_command1="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed111 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 111 --prox_mu 0.01"
eval_command1="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed111 --dataset both --mode fed"

base_command2="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed222 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 222 --prox_mu 0.01"
eval_command2="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed222 --dataset both --mode fed"

base_command3="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed333 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 333 --prox_mu 0.01"
eval_command3="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed333 --dataset both --mode fed"

base_command4="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed444 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 444 --prox_mu 0.01"
eval_command4="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed444 --dataset both --mode fed"

base_command5="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed555 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 555 --prox_mu 0.01"
eval_command5="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed555 --dataset both --mode fed"

base_command6="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed777 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 777 --prox_mu 0.01"
eval_command6="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed777 --dataset both --mode fed"

base_command7="CUDA_VISIBLE_DEVICES=1 python train_fedprox.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --tag both_active0.5_mu0.01_seed888 --log_dir logs1_fedprox --alpha 5.5 --lr_decay -1 --num_epoch 2 --batch_size 8 --dataset both --seed 888 --prox_mu 0.01"
eval_command7="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fedprox --tag both_active0.5_mu0.01_seed888 --dataset both --mode fed"


# 定义日志文件的路径z
log_file="script/fedours_mmd_nodistribn_personal.log"

# 清空日志文件，准备记录新的执行结果
> $log_file

# eval $base_command1 | tee -a $log_file
eval $eval_command1 | tee -a $log_file
# eval $base_command2 | tee -a $log_file
eval $eval_command2 | tee -a $log_file
# eval $base_command3 | tee -a $log_file
eval $eval_command3 | tee -a $log_file
# eval $base_command4 | tee -a $log_file
eval $eval_command4 | tee -a $log_file
# eval $base_command5 | tee -a $log_file
eval $eval_command5 | tee -a $log_file
# eval $base_command6 | tee -a $log_file
eval $eval_command6 | tee -a $log_file
# eval $base_command7 | tee -a $log_file
eval $eval_command7 | tee -a $log_file