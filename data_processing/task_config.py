"""Task order and episode budgets for the eight-family BiRoAD benchmark."""

import argparse
from pathlib import Path


# Each pair is (base roles, swapped roles). Order defines the stored task_id.
TASK_PAIRS = (
    ("bimanual_handover_item_easy_random", "bimanual_handover_item_easy_symmetric_random"),
    ("bimanual_pick_laptop_random", "bimanual_pick_laptop_symmetric_random"),
    ("bimanual_pick_plate_random", "bimanual_pick_plate_symmetric_random"),
    ("bimanual_put_bottle_in_fridge", "bimanual_put_bottle_in_fridge_symmetric"),
    ("bimanual_put_item_in_drawer_random", "bimanual_put_item_in_drawer_symmetric_random"),
    ("bimanual_sweep_to_dustpan_random", "bimanual_sweep_to_dustpan_symmetric_random"),
    ("bimanual_take_tray_out_of_oven", "bimanual_take_tray_out_of_oven_symmetric"),
    ("coordinated_take_shoes_out_of_box", "coordinated_take_shoes_out_of_box_symmetric"),
)
TASK_NAMES = tuple(task for pair in TASK_PAIRS for task in pair)
EPISODE_COUNTS = {"50_50": (50, 50), "95_5": (95, 5)}


def get_task_counts(ratio):
    """Return ordered episode counts, not sampling weights or frame counts."""
    if ratio not in EPISODE_COUNTS:
        raise ValueError(f"Unknown ratio {ratio!r}; choose one of {tuple(EPISODE_COUNTS)}")
    return {
        task: count
        for pair in TASK_PAIRS
        for task, count in zip(pair, EPISODE_COUNTS[ratio])
    }


def relative_path(value):
    """Parse a path relative to the repository root used by the CLI."""
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Use a path relative to the repository root")
    return path
