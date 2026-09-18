# BiRoAD with 3D FlowMatch Actor

Implementation of **BiRoAD: Learning Shared and Role-Adaptive Representations
for Bimanual Manipulation**, built on 3D FlowMatch Actor (3DFA).

This repository provides four experiments: 3DFA and 3DFA + BiRoAD, each trained
with 50:50 or 95:5 base/swapped-role demonstrations. The benchmark contains eight
task families and sixteen task variants. Both methods use the same language
setting and flow-matching loss; BiRoAD adds symmetric and antisymmetric residual
updates to paired arm features.

## Installation

Use Python 3.10+ and a CUDA-enabled PyTorch environment on Linux:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Data conversion and online evaluation additionally require the project's
customized bimanual RLBench, its compatible PyRep, and CoppeliaSim. These task
definitions will be released in a separate repository; its URL, version, and
test-data download instructions are pending. The upstream RLBench tasks alone
do not reproduce this benchmark. Training on prepared Zarr data does not
require the simulator.

The vision encoder uses OpenAI CLIP RN50; the text encoder uses
`openai/clip-vit-base-patch32`. They download pretrained weights on first use.
Launch scripts store caches under `.cache/`. `HF_HOME` and `CLIP_CACHE_DIR` can
select other cache directories; offline mode is optional through the normal
Hugging Face environment variables.

## Data

Run commands from the repository root. All example paths are relative.
See [data preparation](data_processing/README.md) for the raw-data layout.

```bash
bash data_processing/prepare_50_50.sh
bash data_processing/prepare_95_5.sh --splits train
```

Each training preset has 800 episodes: 50+50 or 95+5 per task family. Both use
an independent 50:50 validation set. These ratios count episodes, not keyframes
or sampling weights. Task IDs follow `data_processing/task_config.py`.

## Training

```bash
bash scripts/rlbench/train_3dfa_50_50.sh
bash scripts/rlbench/train_biroad_50_50.sh
bash scripts/rlbench/train_3dfa_95_5.sh
bash scripts/rlbench/train_biroad_95_5.sh
```

Each command starts a separate experiment. Select GPUs with
`CUDA_VISIBLE_DEVICES=0,1`. The global batch size is 256, divided evenly across
the visible GPUs. Training uses 500,000 steps, learning rate 1e-4, and 128×128
images. Set `DATA_PATH` to change the processed-data directory.

Checkpoints are saved to `train_logs/<experiment>/`. `best.pth` is selected by
validation position accuracy (position error below 1 cm), and `last.pth` contains
the resumable training state. To resume:

```bash
CHECKPOINT=train_logs/biroad_95_5/last.pth bash scripts/rlbench/train_biroad_95_5.sh
```

Train and evaluate each checkpoint with the same architecture, BiRoAD options,
and task order.

## Evaluation

Put held-out test episodes under `data/raw/peract2/test/<task_name>/`, using
RLBench's `all_variations/episodes/` or `variation*/episodes/` structure.

```bash
bash online_evaluation_rlbench/scripts/eval_3dfa_50_50.sh
bash online_evaluation_rlbench/scripts/eval_biroad_50_50.sh
bash online_evaluation_rlbench/scripts/eval_3dfa_95_5.sh
bash online_evaluation_rlbench/scripts/eval_biroad_95_5.sh
```

Each command evaluates all sixteen variants, with 50 episodes per variant,
seed 0, at most 25 action steps, and the existing two-attempt action setting.
`CHECKPOINT`, `TEST_DATA_PATH`, and `EVAL_DIR` override the relative paths.
`EVAL_GPU` selects one GPU; `N_WORKERS` controls concurrent task evaluations
(default 2). Video recording is off; append `--save_video true` to enable it.

Results are written to `eval_logs/<experiment>/`. The final `all_results.tsv`
contains base/swapped success rates and Mean, HM, Worst, and Gap. Paired metrics
are computed per task family and then averaged over the eight families. All
sixteen result files are required. To summarize an existing evaluation:

```bash
python online_evaluation_rlbench/collect_results.py --folder eval_logs/biroad_95_5
```

## BiRoAD and ablation options

The training and evaluation entry points forward additional model arguments.
Use the same settings when training and evaluating a checkpoint.

| Argument | Values |
| --- | --- |
| `--use_biroad` | `false` (3DFA), `true` (3DFA + BiRoAD) |
| `--biroad_update_mode` | `residual` (default), `direct` |
| `--biroad_placement` | `early`, `middle`, `late`, `middle_late`, `full` (default) |

`early` applies BiRoAD after language conditioning and after initial observation
conditioning; `middle` follows trajectory–scene feature interaction; `late`
precedes the position and rotation prediction heads. `middle_late` corresponds
to the paper's `middle+late`, and `full` combines all three stages. The 3DFA
scripts disable BiRoAD; the BiRoAD scripts use residual updates at `full`.

For ablations, use a distinct `--exp_log_dir` during training and point
`CHECKPOINT` and `EVAL_DIR` to that experiment during evaluation. No separate
ablation scripts are included.

## Attribution

Based on [3D FlowMatch Actor](https://arxiv.org/abs/2508.11002).
The original copyright and MIT license are retained in [LICENSE](LICENSE).
Source-specific attribution is retained in the adapted modules.
