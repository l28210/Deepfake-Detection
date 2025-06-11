#!/bin/bash

# 定义基础命令
# "FedAvg + aggstrategy1.0"
base_command1="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag nommd_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_seed666 --agg_strategy --agg_factor 1.0 --seed 666"
eval_command1="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag nommd_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_seed666"

# "FedAvg + mmdlast"
base_command2="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmd_distribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_mu 0.01"
eval_command2="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmd_distribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + mmdlast-woBN"
base_command3="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmdnobn_distribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_nobn --mmd_mu 0.01"
eval_command3="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmdnobn_distribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + nodistriBN666"
base_command4="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_aggbn_bothdata_nopersonalized_active0.5_seed666 --seed 666 --nodistribn"
eval_command4="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_aggbn_bothdata_nopersonalized_active0.5_seed666"

# "FedAvg + mmdlast + nodistriBN"
base_command5="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmd_nodistribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_mu 0.01 --nodistribn"
eval_command5="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmd_nodistribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + mmdlast-woBN + nodistriBN"
base_command6="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmdnobn_nodistribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_nobn --mmd_mu 0.01 --nodistribn"
eval_command6="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmdnobn_nodistribn_aggbn_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + mmdlast + aggstrategy1.0"
base_command7="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmd_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_mu 0.01 --agg_strategy --agg_factor 1.0"
eval_command7="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmd_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + mmdlast-woBN + aggstrategy1.0"
base_command8="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmdnobn_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_nobn --mmd_mu 0.01 --agg_strategy --agg_factor 1.0"
eval_command8="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmdnobn_distribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + nodistriBN666 + aggstrategy1.0"
base_command9="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_seed666 --seed 666 --nodistribn --agg_strategy --agg_factor 1.0"
eval_command9="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag nommd_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_seed666"

# "FedAvg + mmdlast + nodistriBN + aggstrategy1.0"
base_command10="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmd_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_mu 0.01 --nodistribn --agg_strategy --agg_factor 1.0"
eval_command10="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmd_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"

# "FedAvg + mmdlast-woBN + nodistriBN + aggstrategy1.0"
base_command11="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.5 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_ours --tag mmdnobn_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666 --seed 666 --mmd --mmd_nobn --mmd_mu 0.01 --nodistribn --agg_strategy --agg_factor 1.0"
eval_command11="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_ours --tag mmdnobn_nodistribn_aggbn_aggstrategy1.0_bothdata_nopersonalized_active0.5_mmdmu0.01_seed666"



# 定义日志文件的路径
log_file="script/fedours.log"

# 清空日志文件，准备记录新的执行结果
> $log_file

eval $base_command1 | tee -a $log_file
eval $eval_command1 | tee -a $log_file
eval $base_command2 | tee -a $log_file
eval $eval_command2 | tee -a $log_file
eval $base_command3 | tee -a $log_file
eval $eval_command3 | tee -a $log_file
eval $base_command4 | tee -a $log_file
eval $eval_command4 | tee -a $log_file
eval $base_command5 | tee -a $log_file
eval $eval_command5 | tee -a $log_file
eval $base_command6 | tee -a $log_file
eval $eval_command6 | tee -a $log_file
eval $base_command7 | tee -a $log_file
eval $eval_command7 | tee -a $log_file
eval $base_command8 | tee -a $log_file
eval $eval_command8 | tee -a $log_file
eval $base_command9 | tee -a $log_file
eval $eval_command9 | tee -a $log_file
eval $base_command10 | tee -a $log_file
eval $eval_command10 | tee -a $log_file
eval $base_command11 | tee -a $log_file
eval $eval_command11 | tee -a $log_file


# CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate 0.1 --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag aggfactor1.0_mmdnobn_nodistribn_aggbn_bothdata_personalized_active0.1_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized --agg_strategy --agg_factor 1.0


















