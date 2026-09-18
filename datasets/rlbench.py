import random
import torch.nn.functional as F

from .base import BaseDataset


from data_processing.task_config import TASK_NAMES


ROLE_SUFFIXES = (" with swapped hands", " using the other hand")


def strip_role_suffix(instr):
    for suffix in ROLE_SUFFIXES:
        if instr.endswith(suffix):
            return instr[:-len(suffix)]
    return instr


class RLBenchDataset(BaseDataset):
    """RLBench dataset."""
    quat_format= 'xyzw'

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4,
        tasks=None,
        all_tasks=None,
        img_size=None,
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size,
            tasks=tasks,
            all_tasks=all_tasks,
            img_size=img_size,
        )

    def _get_task(self, idx):
        if self.all_tasks is None:
            return ["task"] * self.chunk_size
        return [
            self.all_tasks[int(tid)]
            for tid in self.annos['task_id'][idx:idx + self.chunk_size]
        ]


    def _get_instr(self, idx):
        if self.all_tasks is None:
            return ["instruction"] * self.chunk_size
        instrs = []
        for t, v in zip(
            self.annos['task_id'][idx:idx + self.chunk_size],
            self.annos['variation'][idx:idx + self.chunk_size]
        ):
            task_name = self.all_tasks[int(t)]
            instr = random.choice(self._instructions[task_name][str(int(v))])
            instr = strip_role_suffix(instr)
            instrs.append(instr)
        return instrs

    def _get_rgb2d(self, idx):
        if self.camera_inds2d is not None:
            rgb2d = self._get_attr_by_idx(idx, 'rgb', False)[:, self.camera_inds2d]
            if self.img_size is not None and (rgb2d.shape[-1] != self.img_size or rgb2d.shape[-2] != self.img_size):
                b, nc, c, h, w = rgb2d.shape
                rgb2d = F.interpolate(
                    rgb2d.flatten(0, 1).float(), (self.img_size, self.img_size),
                    mode='bilinear', antialias=True
                ).to(rgb2d.dtype).reshape(b, nc, c, self.img_size, self.img_size)
            return rgb2d
        return None

    def _get_extrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        intrinsics = self._get_attr_by_idx(idx, 'intrinsics', True)
        if self.img_size is not None:
            orig_h, orig_w = self.annos['depth'].shape[-2:]
            if orig_h != self.img_size or orig_w != self.img_size:
                scale_h = self.img_size / orig_h
                scale_w = self.img_size / orig_w
                intrinsics = intrinsics.clone()
                intrinsics[..., 0, 0] *= scale_w
                intrinsics[..., 0, 2] *= scale_w
                intrinsics[..., 1, 1] *= scale_h
                intrinsics[..., 1, 2] *= scale_h
        return intrinsics


    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # First detect which copy we fall into
        idx = idx % len(self.indices)
        # and then get the original index from our filtered list
        idx = self.indices[idx]
        if self._actions_only:
            return {"action": self._get_action(idx)}
        ret = {
            "task": self._get_task(idx),  # [str]
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam3d, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam3d, H, W)
            "rgb2d": self._get_rgb2d(idx),  # tensor(n_cam2d, 3, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx),  # tensor(T, 8)
            "extrinsics": self._get_extrinsics(idx),  # tensor(n_cam3d, 4, 4)
            "intrinsics": self._get_intrinsics(idx)  # tensor(n_cam3d, 3, 3)
        }
        return ret


class Peract2Dataset(RLBenchDataset):
    """RLBench dataset under Peract2 setup."""
    tasks = TASK_NAMES
    cameras = ("front", "wrist_left", "wrist_right")
    camera_inds = None
    train_copies = 10
    camera_inds2d = None
