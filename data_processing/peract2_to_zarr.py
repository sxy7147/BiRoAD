"""Convert paired-role RLBench demonstrations to the training Zarr schema."""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm
import zarr

from data_processing.rlbench_utils import (
    image_to_float_array,
    keypoint_discovery,
    store_instructions,
)
from data_processing.task_config import EPISODE_COUNTS, get_task_counts, relative_path


DEPTH_SCALE = 2**24 - 1
CAMERAS = ("front", "wrist_left", "wrist_right")
NUM_ARMS = 2


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=relative_path, default=Path("data/raw/peract2"))
    parser.add_argument("--output-dir", type=relative_path, default=Path("data/processed/peract2"))
    parser.add_argument("--ratio", choices=tuple(EPISODE_COUNTS), default="50_50",
                        help="Base/swapped episode counts for training; validation is always 50:50")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--image-size", type=int, default=128,
                        help="Expected raw image size; images are not resized")
    parser.add_argument("--num-cameras", type=int, choices=(1, 3), default=3)
    parser.add_argument("--store-episode-metadata", action=argparse.BooleanOptionalAction,
                        default=True, help="Store episode_id and step_idx")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild an existing output directory")
    return parser.parse_args(argv)


def build_split_plans(ratio, splits, output_dir):
    """Both training presets point to the same balanced validation split."""
    get_task_counts(ratio)
    plans = []
    for split in dict.fromkeys(splits):
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split: {split}")
        split_ratio = ratio if split == "train" else "50_50"
        plans.append((split, Path(output_dir) / f"{split}_{split_ratio}", get_task_counts(split_ratio)))
    return plans


def select_episodes(episodes_dir, count):
    """Select the requested prefix in lexicographic order, failing on shortages."""
    episodes_dir = Path(episodes_dir)
    if count <= 0:
        raise ValueError("Episode count must be positive")
    episodes = sorted(
        (path for path in episodes_dir.iterdir() if path.is_dir() and path.name.startswith("ep")),
        key=lambda path: path.name,
    )
    if len(episodes) < count:
        raise ValueError(f"{episodes_dir}: requested {count} episodes, found {len(episodes)}")
    selected = episodes[:count]
    for episode in selected:
        for filename in ("low_dim_obs.pkl", "variation_number.pkl", "variation_descriptions.pkl"):
            if not (episode / filename).is_file():
                raise FileNotFoundError(episode / filename)
    return selected


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def read_episode(episode_dir, task_id, episode_id, cameras, image_size, store_episode_metadata):
    """Read keyframe transitions in the original camera and left/right arm order."""
    demo = load_pickle(episode_dir / "low_dim_obs.pkl")
    if len(demo) < 2:
        raise ValueError(f"{episode_dir}: an episode needs at least two observations")
    keyframe_indices = [0] + keypoint_discovery(demo, bimanual=True)
    observation_indices = keyframe_indices[:-1]
    if not observation_indices:
        raise ValueError(f"{episode_dir}: no keyframe transitions found")

    rgb_frames, depth_frames = [], []
    for frame in observation_indices:
        camera_rgb, camera_depth = [], []
        for camera in cameras:
            with Image.open(episode_dir / f"{camera}_rgb/rgb_{frame:04d}.png") as image:
                rgb = np.array(image)
            with Image.open(episode_dir / f"{camera}_depth/depth_{frame:04d}.png") as image:
                depth = image_to_float_array(image, DEPTH_SCALE)
            if rgb.shape != (image_size, image_size, 3) or depth.shape != (image_size, image_size):
                raise ValueError(f"{episode_dir}/{camera}: image size must be {image_size}x{image_size}")
            near = demo[frame].misc[f"{camera}_camera_near"]
            far = demo[frame].misc[f"{camera}_camera_far"]
            camera_rgb.append(rgb)
            camera_depth.append(near + depth * (far - near))
        rgb_frames.append(np.stack(camera_rgb))
        depth_frames.append(np.stack(camera_depth).astype(np.float16))

    poses = np.array([
        [np.concatenate((getattr(demo[frame], arm).gripper_pose,
                         [getattr(demo[frame], arm).gripper_open]))
         for arm in ("left", "right")]
        for frame in keyframe_indices
    ], dtype=np.float32)
    joint_states = np.array([
        [np.concatenate((getattr(demo[frame], arm).joint_positions,
                         [getattr(demo[frame], arm).gripper_open]))
         for arm in ("left", "right")]
        for frame in keyframe_indices
    ], dtype=np.float32)

    # The history contains two previous poses and the current pose. Repeat the
    # initial pose at the beginning of each episode, exactly as in training.
    current = poses[:-1]
    previous = np.concatenate((current[:1], current[:-1]))
    previous_previous = np.concatenate((previous[:1], previous[:-1]))
    num_steps = len(observation_indices)
    variation = int(load_pickle(episode_dir / "variation_number.pkl"))
    if not 0 <= variation <= np.iinfo(np.uint8).max:
        raise ValueError(f"{episode_dir}: variation {variation} does not fit the uint8 schema")
    fields = {
        "rgb": np.stack(rgb_frames).transpose(0, 1, 4, 2, 3),
        "depth": np.stack(depth_frames),
        "proprioception": np.stack((previous_previous, previous, current), axis=1),
        "action": poses[1:].reshape(num_steps, 1, NUM_ARMS, 8),
        "proprioception_joints": joint_states[:-1].reshape(num_steps, 1, NUM_ARMS, 8),
        "action_joints": joint_states[1:].reshape(num_steps, 1, NUM_ARMS, 8),
        "extrinsics": np.array([
            [demo[frame].misc[f"{camera}_camera_extrinsics"] for camera in cameras]
            for frame in observation_indices
        ], dtype=np.float16),
        "intrinsics": np.array([
            [demo[frame].misc[f"{camera}_camera_intrinsics"] for camera in cameras]
            for frame in observation_indices
        ], dtype=np.float16),
        "task_id": np.full(num_steps, task_id, dtype=np.uint8),
        "variation": np.full(num_steps, variation, dtype=np.uint8),
    }
    if store_episode_metadata:
        fields["episode_id"] = np.full(num_steps, episode_id, dtype=np.uint32)
        fields["step_idx"] = np.arange(num_steps, dtype=np.uint16)
    return fields


def convert_split(raw_data_dir, output_dir, split, task_counts, image_size=128,
                  num_cameras=3, store_episode_metadata=True, overwrite=False):
    """Write one split, retaining a previous dataset if conversion fails."""
    raw_data_dir, output_dir = Path(raw_data_dir), Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already exists; choose another split or use --overwrite")
    if image_size <= 0 or num_cameras not in (1, 3):
        raise ValueError("image_size must be positive and num_cameras must be 1 or 3")
    if not task_counts or len(task_counts) > 256:
        raise ValueError("Provide between 1 and 256 tasks for the uint8 task_id schema")
    if output_dir.is_symlink() or raw_data_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Output must not be a symlink or contain the raw dataset")
    cameras = CAMERAS if num_cameras == 3 else CAMERAS[:1]
    selected = {
        task: select_episodes(raw_data_dir / split / task / "all_variations/episodes", count)
        for task, count in task_counts.items()
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Convert into a sibling temporary directory before publishing the result.
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staged_dir = Path(temporary) / "dataset"
        staged_dir.mkdir()
        group = zarr.open_group(str(staged_dir / f"{split}.zarr"), mode="w")
        group.attrs["task_names"] = list(task_counts)
        compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
        episode_id = 0
        for task_id, (task, episodes) in enumerate(selected.items()):
            for episode in tqdm(episodes, desc=f"{split}/{task}"):
                fields = read_episode(episode, task_id, episode_id, cameras, image_size, store_episode_metadata)
                for name, values in fields.items():
                    if name not in group:
                        group.create_dataset(name, shape=(0,) + values.shape[1:],
                                             chunks=(1,) + values.shape[1:], dtype=values.dtype,
                                             compressor=compressor)
                    group[name].append(values)
                episode_id += 1
        instructions = store_instructions(raw_data_dir, list(task_counts), splits=[split])
        (staged_dir / "instructions.json").write_text(json.dumps(instructions, indent=2) + "\n")
        num_transitions = len(group["action"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staged_dir.rename(output_dir)
    print(f"Wrote {output_dir}: {episode_id} episodes, {num_transitions} keyframe transitions")
    return output_dir


def main(argv=None):
    args = parse_arguments(argv)
    plans = build_split_plans(args.ratio, args.splits, args.output_dir)
    if not args.overwrite:
        for _, output_dir, _ in plans:
            if output_dir.exists():
                raise FileExistsError(f"{output_dir} already exists; choose another split or use --overwrite")
    for split, output_dir, task_counts in plans:
        convert_split(args.raw_data_dir, output_dir, split, task_counts,
                      image_size=args.image_size, num_cameras=args.num_cameras,
                      store_episode_metadata=args.store_episode_metadata, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
