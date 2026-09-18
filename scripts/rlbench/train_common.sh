#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

model=$1
ratio=$2
shift 2
experiment="${model}_${ratio}"
data_dir=${DATA_PATH:-data/processed/peract2}
export HF_HOME=${HF_HOME:-.cache/huggingface}
export CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-.cache/clip}

num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( num_gpus < 1 || 256 % num_gpus != 0 )); then
    echo "Use a GPU count that divides the global batch size of 256." >&2
    exit 1
fi
batch_size=$((256 / num_gpus))
task_names=$(python -c 'from data_processing.task_config import TASK_NAMES; print(" ".join(TASK_NAMES))')
read -r -a tasks <<< "$task_names"
use_biroad=false
if [[ "$model" == biroad ]]; then
    use_biroad=true
fi

exec torchrun --standalone --nproc_per_node "$num_gpus" main.py \
    --train_data_dir "$data_dir/train_$ratio/train.zarr" \
    --eval_data_dir "$data_dir/val_50_50/val.zarr" \
    --train_instructions "$data_dir/train_$ratio/instructions.json" \
    --val_instructions "$data_dir/val_50_50/instructions.json" \
    --tasks "${tasks[@]}" --all_tasks "${tasks[@]}" \
    --dataset Peract2_3dfront_3dwrist \
    --custom_img_size 128 --num_workers 8 \
    --batch_size "$batch_size" --batch_size_val 64 \
    --chunk_size 1 --memory_limit 16 \
    --exp_log_dir "$experiment" --run_log_dir . \
    --checkpoint "${CHECKPOINT:-none}" \
    --val_freq 10000 --num_best_checkpoints 3 --train_iters 500000 \
    --lr 1e-4 --backbone_lr 1e-6 --lr_scheduler constant --wd 1e-10 \
    --use_compile false --use_ema false --lv2_batch_size 1 \
    --model_type denoise3d --bimanual true --keypose_only true --pre_tokenize true \
    --backbone clip --finetune_backbone false --finetune_text_encoder false \
    --fps_subsampling_factor 4 --embedding_dim 120 --num_attn_heads 8 \
    --num_vis_instr_attn_layers 3 --num_history 3 --num_shared_attn_layers 4 \
    --workspace_normalizer_buffer 0.05 --relative_action false \
    --rotation_format quat_xyzw --denoise_timesteps 5 --denoise_model rectified_flow \
    --no_hand_embed false \
    --use_biroad "$use_biroad" \
    --biroad_update_mode residual --biroad_placement full \
    "$@"
