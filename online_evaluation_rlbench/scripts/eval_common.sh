#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

model=$1
ratio=$2
shift 2
experiment="${model}_${ratio}"
checkpoint=${CHECKPOINT:-train_logs/$experiment/best.pth}
data_dir=${TEST_DATA_PATH:-data/raw/peract2/test}
output_dir=${EVAL_DIR:-eval_logs/$experiment}
num_workers=${N_WORKERS:-2}
if (( num_workers < 1 )); then
    echo "N_WORKERS must be at least 1." >&2
    exit 1
fi
gpu_ids=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-${gpu_ids%%,*}}
export HF_HOME=${HF_HOME:-.cache/huggingface}
export CLIP_CACHE_DIR=${CLIP_CACHE_DIR:-.cache/clip}
task_names=$(python -c 'from data_processing.task_config import TASK_NAMES; print(" ".join(TASK_NAMES))')
read -r -a tasks <<< "$task_names"
use_biroad=false
if [[ "$model" == biroad ]]; then
    use_biroad=true
fi

pids=()
for task in "${tasks[@]}"; do
    python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint "$checkpoint" --task "$task" --data_dir "$data_dir" \
        --output_file "$output_dir/$task/eval.json" \
        --dataset Peract2_3dfront_3dwrist --image_size 128,128 \
        --num_demos 50 --seed 0 --max_tries 2 --max_steps 25 \
        --headless true --collision_checking false \
        --save_video false --video_dir "$output_dir/$task/videos" \
        --model_type denoise3d --bimanual true --prediction_len 1 \
        --backbone clip --fps_subsampling_factor 4 \
        --embedding_dim 120 --num_attn_heads 8 --num_vis_instr_attn_layers 3 \
        --num_history 3 --num_shared_attn_layers 4 --relative_action false \
        --rotation_format quat_xyzw --denoise_timesteps 5 --denoise_model rectified_flow \
        --no_hand_embed false \
        --use_biroad "$use_biroad" \
        --biroad_update_mode residual --biroad_placement full \
        "$@" &
    pids+=("$!")
    if (( ${#pids[@]} >= num_workers )); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
    fi
done
if (( ${#pids[@]} > 0 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
fi
python online_evaluation_rlbench/collect_results.py --folder "$output_dir"
