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

# use model parallel if you have multiple small gpus on a single node, will be slower
# tune micro_batch_size to be the largest value that does not cause OOM

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
output_dir='../exp/ltuas_ft_toy_low_resource/'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

python ../finetune_low_resource.py \
    --base_model '../../../pretrained_mdls/ltuas_long_noqa_a6.bin' \
    --data_path '../../../openasqa/data/openasqa_toy.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 1 \
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

# should see something like this
#{'loss': 0.6051, 'learning_rate': 2e-05, 'epoch': 0.29}
#{'loss': 0.5754, 'learning_rate': 4e-05, 'epoch': 0.58}
#{'loss': 0.5488, 'learning_rate': 6e-05, 'epoch': 0.88}
#{'train_runtime': 2219.2005, 'train_samples_per_second': 3.951, 'train_steps_per_second': 0.015, 'train_loss': 0.5715190382564769, 'epoch': 0.99}
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [36:48<00:00, 64.95s/it]
