#!/bin/bash

# 定义active_rate和seed的数组
active_rates=(0.5)
seeds=(111 222 333 444 666 777 888 999)

log_file="script/fedmmd_randactive5.log"
# 外层循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    # 内层循环遍历agg_factor
    for seed in "${seeds[@]}"; do
        # 构建训练命令
        base_command="CUDA_VISIBLE_DEVICES=0 python train_fed.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs1_fedmmd --tag mmd_bothdata_active${active_rate}_mmdmu0.01_seed${seed} --seed ${seed} --mmd --mmd_mu 0.01"
        
        # 构建评估命令
        eval_command="CUDA_VISIBLE_DEVICES=0 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs1_fedmmd --tag mmd_bothdata_active${active_rate}_mmdmu0.01_seed${seed} --savepcd"
        
        # 打印生成的命令
        echo "Training Command:"
        echo $base_command
        echo "Evaluation Command:"
        echo $eval_command
        echo ""  # 打印空行以分隔不同命令
        
        # 执行训练命令
        echo "Executing training command..."
        eval $base_command | tee -a $log_file
        
        # 等待命令执行完成
        echo "Waiting for training to finish..."
        wait
        
        # 执行评估命令
        echo "Executing evaluation command..."
        eval $eval_command | tee -a $log_file
    done
done















