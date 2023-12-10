#!/bin/bash
#SBATCH -J alm
#SBATCH -o ./log/%j_alm.txt
#SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a6
#SBATCH --ntasks-per-node=32
#SBATCH --mem=470000
#SBATCH --exclusive

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
output_dir='../exp/ltuas_ft_toy/'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=4 --master_port=1234 ../finetune.py \
    --base_model '../../../pretrained_mdls/ltuas_long_noqa_a6.bin' \
    --data_path '../../../openasqa/data/openasqa_toy.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 2e-4 \
    --cutoff_len 100 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 30 \
    --trainable_params all

pkill -f wandb

#{'loss': 0.6406, 'learning_rate': 2e-05, 'epoch': 0.29}
#{'loss': 0.5767, 'learning_rate': 4e-05, 'epoch': 0.58}
#{'loss': 0.5612, 'learning_rate': 6e-05, 'epoch': 0.87}