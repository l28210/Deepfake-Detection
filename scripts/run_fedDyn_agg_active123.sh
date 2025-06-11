#!/bin/bash
# FedDyn加入DGA（agg_factors控制DGA的权重）
active_rates=(0.1 0.2 0.3)
seeds=(800)
agg_factors=(1.0 0.2 0.4 0.6 0.8 1.2 1.4 1.6 1.8 2.0)


log_file="script/fedDyn_agg_active123.log"

# 循环遍历active_rate
for active_rate in "${active_rates[@]}"; do
    for seed in "${seeds[@]}"; do
        for agg_factor in "${agg_factors[@]}"; do
            # baseline
            baseline_train="CUDA_VISIBLE_DEVICES=2 python train_fedDyn.py --num_clients 10 --active_rate $active_rate --num_rounds 20 --num_epochs 2 --batch_size 8 --lr_decay -1 --alpha 5.5 --dataset both --log_dir logs_fedDyn_baseline_active123 --tag fedDyn_agg${agg_factor}_active${active_rate}_seed${seed} --seed ${seed} --agg_strategy --agg_factor ${agg_factor}"
            # baseline_eval="CUDA_VISIBLE_DEVICES=2 python eval_central.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_fedDyn_baseline_active123 --tag fedDyn_agg${agg_factor}_active${active_rate}_seed${seed} --mode fed"
            baseline_eval="CUDA_VISIBLE_DEVICES=2 python eval_fed.py --batch_size 8 --threshold 8e-3 --dataset both --log_dir logs_fedDyn_baseline_active123 --tag fedDyn_agg${agg_factor}_active${active_rate}_seed${seed}"

            
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
done

