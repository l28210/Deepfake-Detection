#!/bin/bash

# 定义基础命令
base_command="CUDA_VISIBLE_DEVICES=1 python train_fednorm.py --num_clients 10 --num_rounds 20 --alpha 5.5 --lr_decay -1 --num_epoch 2 --log_dir logs1_fednorm --batch_size 8 --dataset both"

eval_command="CUDA_VISIBLE_DEVICES=1 python eval_central.py --batch_size 8 --threshold 8e-3 --log_dir logs1_fednorm --dataset both --mode fed --savepcd"

active_rate=0.5

# 定义要遍历的seed值数组
# seeds=(444 555 777 888 999 114 514) # 你可以在这里添加更多的seed值
seeds=(444 555 999 114 514)

# 定义日志文件的路径
log_file="script/fednorm_active${active_rate}_multiple_seeds.log"

# 清空日志文件，准备记录新的执行结果
> $log_file

# 遍历数组中的每个seed值
for seed in "${seeds[@]}"
do
    # 构建新的命令行
    new_command="${base_command} --active_rate ${active_rate} --tag both_active${active_rate}_seed${seed}  --seed ${seed}"

    new_eval_command="${eval_command} --tag both_active${active_rate}_seed${seed}"

    # 打印即将执行的命令
    # echo "Executing command for seed ${seed}: ${new_command}" | tee -a $log_file

    # 执行命令，并将输出重定向到日志文件
    # eval $new_command | tee -a $log_file

    echo "Executing eval command for seed ${seed}: ${new_eval_command}" | tee -a $log_file
    eval $new_eval_command | tee -a $log_file

    # 打印命令执行完成的状态
    echo "Command executed with seed ${seed}. Check log for details." | tee -a $log_file
    echo "----------------------------------------" | tee -a $log_file
done