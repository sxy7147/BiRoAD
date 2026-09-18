# PerAct2 data preparation

Eight task families with two training presets:

| Preset | Base / swapped episodes per task | Total |
| --- | --- | --- |
| `50_50` | 50 / 50 | 800 |
| `95_5` | 95 / 5 | 800 |

Both use an independent 50:50 validation set. These are episode counts;
keyframe counts can differ. The task names and order are in `task_config.py`.

## Input and environment

Use the RLBench/PyRep environment that collected the demonstrations:

```bash
python -m pip install -r data_processing/requirements.txt
```

Paths are relative to the repository root. Raw episodes should be under:

```text
data/raw/peract2/train/<task_name>/all_variations/episodes/episode0/
data/raw/peract2/val/<task_name>/all_variations/episodes/episode0/
```

Each episode needs `low_dim_obs.pkl`, `variation_number.pkl`,
`variation_descriptions.pkl`, and the RGB/depth folders for `front`,
`wrist_left`, `wrist_right`. Default image size is 128×128; no resizing is done.

## Run

```bash
# Prepare 50:50 training data and the shared validation data.
bash data_processing/prepare_50_50.sh
# Prepare 95:5 training data; the validation data are already available.
bash data_processing/prepare_95_5.sh --splits train
```

Outputs go to `data/processed/peract2/{train_50_50,train_95_5,val_50_50}/`.
Each directory contains `train.zarr` or `val.zarr`, plus `instructions.json`.
Use `--raw-data-dir ../datasets/peract2` or `--output-dir data/processed/peract2`
to choose other relative paths. Existing outputs cause an error; add
`--overwrite` only when you want to rebuild them. Use `--splits val` to prepare
validation alone. `--help` lists the remaining options.

Episodes are selected in lexicographic order; insufficient episodes cause an
error. Camera order is front/left wrist/right wrist; arm order is left/right.
The 16 task names are stored in the Zarr `task_names` attribute. Training must
use that order. Optional `episode_id` and `step_idx` fields can be disabled with
`--no-store-episode-metadata`.
