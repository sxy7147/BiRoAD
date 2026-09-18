"""Online evaluation script on RLBench."""

import os
import sys

import argparse
import random
from pathlib import Path
import json

# Add the root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none, round_floats


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for RLBench evaluation")
    # Tuples: (name, type, default)
    arguments = [
        # Evaluation arguments
        ('checkpoint', str_none, None),
        ('task', str, "bimanual_handover_item_easy_random"),
        ('max_tries', int, 2),
        ('max_steps', int, 25),
        ('headless', str2bool, True),
        ('collision_checking', str2bool, False),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, "data/raw/peract2/test"),
        ('eval_instructions', str_none, None),
        ('dataset', str, "Peract2_3dfront_3dwrist"),
        ('image_size', str, "128,128"),
        # Logging arguments
        ('output_file', Path, "eval_logs/eval.json"),
        ('num_demos', int, 50),
        ('from_episode_number', int, 0),
        ('save_video', str2bool, False),
        ('video_dir', Path, "eval_logs/videos"),
        ('video_resolution', str, "256,256"),
        ('video_camera', str_none, None),
        ('video_freq', int, 5),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, True),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('fps_subsampling_factor', int, 4),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 3),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 5),
        ('denoise_model', str, "rectified_flow"),
        ('no_hand_embed', str2bool, False),
        ('use_biroad', str2bool, False),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    parser.add_argument(
        '--biroad_update_mode', choices=('residual', 'direct'), default='residual'
    )
    parser.add_argument(
        '--biroad_placement',
        choices=('early', 'middle', 'late', 'middle_late', 'full'),
        default='full',
    )
    return parser.parse_args()


def load_models(args):
    print("Loading model from", args.checkpoint, flush=True)

    model_class = fetch_model_class(args.model_type)
    model = model_class(
        backbone=args.backbone,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        no_hand_embed=args.no_hand_embed,
        use_biroad=args.use_biroad,
        biroad_update_mode=args.biroad_update_mode,
        biroad_placement=args.biroad_placement,
    )

    # Load model weights
    model_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )

    weights = {
        key.removeprefix("module."): value
        for key, value in model_dict["weight"].items()
    }
    model.load_state_dict(weights, strict=True)

    model.eval()

    return model.cuda()


if __name__ == "__main__":
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    # Save results here
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create video directory if saving videos
    if args.save_video and args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner

    # Dataset class (for getting cameras and tasks/variations)
    dataset_class = fetch_dataset_class(args.dataset)

    # Load models
    model = load_models(args)

    # Evaluate - reload environment for each task (crashes otherwise)
    task_success_rates = {}
    for task_str in [args.task]:

        # Seeds - re-seed for each task
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Load RLBench environment
        env = RLBenchEnv(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=dataset_class.cameras,
            collision_checking=bool(args.collision_checking)
        )

        # Actioner (runs the policy online)
        actioner_kwargs = {
            "backbone": args.backbone,
            "task_str": args.task,
        }
        if args.bimanual:
            actioner_kwargs["eval_instructions"] = args.eval_instructions
        actioner = Actioner(model, **actioner_kwargs)

        # Evaluate
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history,
            num_demos=int(args.num_demos),
            from_episode_number=int(args.from_episode_number),
            save_video=args.save_video,
            video_dir=args.video_dir,
            video_resolution=tuple(int(x) for x in args.video_resolution.split(",")),
            video_camera=args.video_camera,
            video_freq=args.video_freq
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(task_success_rates, f, indent=4)
