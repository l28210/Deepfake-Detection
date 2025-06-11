#!/bin/bash

active_rates=(0.2 0.3)
seeds=(222 333 444 555 666 777 888 999)

log_file="script/fedmmd_noagg_active123.log"

# 循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    # 循环遍历seed
    for seed in "${seeds[@]}"; do
        # baseline
        baseline_train="CUDA_VISIBLE_DEVICES=0 python train_fed.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_fedmmd_agg --tag noagg_mmd0.01_active${active_rate}_seed${seed} --seed ${seed} --mmd --mmd_mu 0.01"
        baseline_eval="CUDA_VISIBLE_DEVICES=0 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_fedmmd_agg --tag noagg_mmd0.01_active${active_rate}_seed${seed}"
        
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
done
