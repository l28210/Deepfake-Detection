#!/bin/bash
# FedProx 不包括DGA

active_rates=(0.1 0.2 0.3)

log_file="script/fedprox_base_active123.log"

# 循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    # baseline
    baseline_train="CUDA_VISIBLE_DEVICES=0 python train_fedprox.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_baseline_active123 --tag fedprox_active${active_rate}_seed111 --seed 111"
    baseline_eval="CUDA_VISIBLE_DEVICES=0 python eval_central.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_baseline_active123 --tag fedprox_active${active_rate}_seed111 --mode fed"
    
    # 执行训练命令
    echo "Executing training command..."
    eval $baseline_train | tee -a $log_file
    
    # 等待命令执行完成
    echo "Waiting for training to finish..."
    wait
    
    # 执行评估命令
    echo "Executing evaluation command..."
    eval $baseline_eval | tee -a $log_file
done

