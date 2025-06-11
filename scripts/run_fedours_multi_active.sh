#!/bin/bash

# 定义基础命令
base_command0="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.1 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.1_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized"
eval_command0="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.1_mmdmu0.01_seed111"

base_command1="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.2 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.2_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized"
eval_command1="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.2_mmdmu0.01_seed111"

base_command2="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.3 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.3_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized"
eval_command2="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.3_mmdmu0.01_seed111"

base_command3="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate 0.4 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.4_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized"
eval_command3="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs2_ours --tag mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.4_mmdmu0.01_seed111"


# 定义日志文件的路径
log_file="script/fedours_multiactive.log"

# 清空日志文件，准备记录新的执行结果
> $log_file

eval $base_command0 | tee -a $log_file
eval $eval_command0 | tee -a $log_file
# eval $base_command1 | tee -a $log_file
# eval $eval_command1 | tee -a $log_file
# eval $base_command2 | tee -a $log_file
# eval $eval_command2 | tee -a $log_file
# eval $base_command3 | tee -a $log_file
# eval $eval_command3 | tee -a $log_file
# eval $base_command4 | tee -a $log_file
# eval $eval_command4 | tee -a $log_file
# eval $base_command5 | tee -a $log_file
# eval $eval_command5 | tee -a $log_file
# eval $base_command6 | tee -a $log_file
# eval $eval_command6 | tee -a $log_file
# eval $base_command7 | tee -a $log_file
# eval $eval_command7 | tee -a $log_file
# eval $base_command8 | tee -a $log_file
# eval $eval_command8 | tee -a $log_file
# eval $base_command9 | tee -a $log_file
# eval $eval_command9 | tee -a $log_file
# eval $base_command10 | tee -a $log_file
# eval $eval_command10 | tee -a $log_file
# eval $base_command11 | tee -a $log_file
# eval $eval_command11 | tee -a $log_file



















