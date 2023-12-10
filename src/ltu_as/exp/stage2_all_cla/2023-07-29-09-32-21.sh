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

export TRANSFORMERS_CACHE=/data/sls/scratch/yuangong/audiollm/hf_cache/
export HF_DATASETS_CACHE=/data/sls/scratch/yuangong/audiollm/hf_cache/
output_dir='../exp_after_sub/formal_joint_all_cla_unified_feat'
mkdir $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=4 --master_port=1234 ../finetune.py \
    --base_model '/data/sls/scratch/yuangong/audiollm/src/llm/ltu_e/exp_after_sub/formal_joint_proj_cla_unified_feat/checkpoint-16000/pytorch_model.bin' \
    --data_path '/data/sls/scratch/yuangong/audiollm/src/data/prep_data_ltue/speech_qa/close_end/datafiles/combine_paralinguistic_at_music_whisper.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 6 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 128 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 1000 \
    --trainable_params all

pkill -f wandb