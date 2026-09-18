from .rlbench import RLBenchDepth2Cloud


def fetch_depth2cloud(dataset_name, img_size=128):
    return RLBenchDepth2Cloud((img_size, img_size))
