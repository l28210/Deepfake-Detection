#!/bin/bash

# 定义active_rate和agg_factor的数组
active_rates=(0.5)
agg_factors=(0.6 0.4 0.2)

log_file="script/fedours_multiactive_agg.log"
# 外层循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    # 内层循环遍历agg_factor
    for agg_factor in "${agg_factors[@]}"; do
        # 构建训练命令
        base_command="CUDA_VISIBLE_DEVICES=2 python train_fed.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs2_ours --tag aggfactor${agg_factor}_mmdnobn_nodistribn_aggbn_bothdata_personalized_active${active_rate}_mmdmu0.01_seed111 --seed 111 --mmd --mmd_mu 0.01 --mmd_nobn --nodistribn --personalized --agg_strategy --agg_factor $agg_factor"
        
        # 构建评估命令
        eval_command="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs2_ours --tag aggfactor${agg_factor}_mmdnobn_nodistribn_aggbn_bothdata_personalized_active${active_rate}_mmdmu0.01_seed111"
        
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















