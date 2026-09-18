from functools import partial

from .rlbench import RLBenchDataPreprocessor


def fetch_data_preprocessor(dataset_name, img_size=128):
    return partial(RLBenchDataPreprocessor, orig_imsize=img_size)
