#!/bin/bash
# Ours方法的 包括和不包括DGA agg_factors控制DGA的权重

active_rates=(0.3)
agg_factors=(1.0 1.2 1.4 1.6 1.8 2.0 0.8 0.6 0.4 0.2)

log_file="script/fedours_logs_agg_active123.log"
# 循环遍历agg_factor
for agg_factor in "${agg_factors[@]}"; do
    # 循环遍历active_rate
    for active_rate in "${active_rates[@]}"; do
        # baseline
        if [[ `expr $agg_factor == 1.0`  == 1 ]]; then
            baseline_train="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_ours_agg --tag noagg_nommd_nodistribn_personalized_active${active_rate}_seed111 --nodistribn --personalized --seed 111"
            baseline_eval="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_ours_agg --tag noagg_nommd_nodistribn_personalized_active${active_rate}_seed111"
            
            # 执行训练命令
            echo "Executing training command..."
            eval $baseline_train | tee -a $log_file
            
            # 等待命令执行完成
            echo "Waiting for training to finish..."
            wait
            
            # 执行评估命令
            echo "Executing evaluation command..."
            eval $baseline_eval | tee -a $log_file
        fi

        # 构建训练命令
        base_command="CUDA_VISIBLE_DEVICES=1 python train_fed.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_ours_agg --tag agg${agg_factor}_nommd_nodistribn_personalized_active${active_rate}_seed111 --seed 111 --nodistribn --personalized --agg_strategy --agg_factor $agg_factor"
        
        # 构建评估命令
        eval_command="CUDA_VISIBLE_DEVICES=1 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_ours_agg --tag agg${agg_factor}_nommd_nodistribn_personalized_active${active_rate}_seed111"
        
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
