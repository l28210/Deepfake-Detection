#!/bin/bash

CUDA_VISIBLE_DEVICES=2
num_epochs=20
batch_size=32
lr=0.001
lr_decay=0.05
log_dir="logs_central"
seeds=(114514)
log_file="script/central.log"
data_folder="data"
LARGE_KERNEL_CONV_IMPL="lib/RepLKNet-pytorch/cutlass/examples/19_large_depthwise_conv2d_torch_extension/"
data_set="archive"
CPUs=16

for seed in "${seeds[@]}"; do
    baseline_train="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_central.py --num_epochs ${num_epochs} --batch_size ${batch_size} --lr ${lr} --lr_decay ${lr_decay} --folder_data ${data_folder}  --LARGE_KERNEL_CONV_IMPL ${LARGE_KERNEL_CONV_IMPL} --log_dir ${log_dir} --tag central_seed${seed} --dataset ${data_set} --seed ${seed} --CPUs ${CPUs}"
    
    baseline_eval="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python eval_central.py --batch_size ${batch_size} --log_dir ${log_dir} --dataset ${data_set} --CPUs ${CPUs} "

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