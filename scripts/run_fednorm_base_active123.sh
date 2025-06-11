#!/bin/bash
# FedNoem不包括DGA
active_rates=(0.1 0.2 0.3)
seeds=(555)

log_file="script/fednorm_logs_active123.log"

# 循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    for seed in "${seeds[@]}"; do
        # baseline
        baseline_train="CUDA_VISIBLE_DEVICES=1 python train_fednorm.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_baseline_active123 --tag fednorm_active${active_rate}_seed${seed} --seed $seed"
        baseline_eval="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_baseline_active123 --tag fednorm_active${active_rate}_seed${seed} --mode fed"
        
        # 执行训练命令
        # echo "Executing training command..."
        # eval $baseline_train | tee -a $log_file
        
        # # 等待命令执行完成
        # echo "Waiting for training to finish..."
        # wait
        
        # 执行评估命令
        echo "Executing evaluation command..."
        eval $baseline_eval | tee -a $log_file
    done
done
