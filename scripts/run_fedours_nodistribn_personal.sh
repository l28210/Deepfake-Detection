#!/bin/bash

# 定义基础命令
base_command1="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_8head_bothdata_personalized_active0.5_seed111 --seed 111 --nodistribn --personalized"
eval_command1="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_8head_bothdata_personalized_active0.5_seed111"

base_command2="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag agg1.0_nommd_nodistribn_8head_bothdata_personalized_active0.5_seed111 --seed 111 --nodistribn --personalized --agg_strategy --agg_factor 1.0"
eval_command2="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag agg1.0_nommd_nodistribn_8head_bothdata_personalized_active0.5_seed111"


# 定义日志文件的路径
log_file="script/fedours_nodistribn_personal.log"

# 清空日志文件，准备记录新的执行结果
> $log_file

eval $base_command1 | tee -a $log_file
eval $eval_command1 | tee -a $log_file
eval $base_command2 | tee -a $log_file
eval $eval_command2 | tee -a $log_file
