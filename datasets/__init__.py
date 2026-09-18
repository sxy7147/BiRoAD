from .rlbench import Peract2Dataset


def fetch_dataset_class(dataset_name):
    if dataset_name != "Peract2_3dfront_3dwrist":
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return Peract2Dataset
