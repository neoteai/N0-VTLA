# Post-training on your own robot data

The reference workflow has four stages: convert the raw episodes, compute normalization
statistics, configure the pretrained checkpoint, and launch task-level post-training. The
reference config is `vtla_tactile_posttrain`.

This recipe expects an 𝒩₀-VTLA-compatible pretrained checkpoint that already contains
the tactile encoder, tactile predictor, and projection parameters. A plain base-VLA checkpoint does not
contain those parameters and would leave them uninitialized.

### 1. Convert episodes to canonical LeRobot

The canonical representation uses a fixed 32-dimensional state/action layout:

| Dimensions | Meaning |
| --- | --- |
| `0:3` | left-arm EEF position |
| `3:9` | left-arm EEF rotation in 6D representation |
| `9` | left gripper |
| `10:13` | right-arm EEF position |
| `13:19` | right-arm EEF rotation in 6D representation |
| `19` | right gripper |
| `20:32` | reserved padding |

Single-arm datasets use the first ten dimensions and mask the remaining dimensions. Camera and
tactile slots are also fixed. An unavailable view must be represented by a zero placeholder
with a false mask; do not substitute a different physical camera.

Convert Flexiv episodes:

```bash
python scripts/convert_canonical_data.py \
  /path/to/raw_episodes \
  /path/to/datasets/canonical_tactile_task \
  --robot flexiv \
  --task "Perform the task"
```

Convert ALOHA episodes with the default dual-arm EEF representation:

```bash
python scripts/convert_canonical_data.py \
  /path/to/raw_episodes \
  /path/to/datasets/canonical_tactile_task \
  --robot aloha \
  --task "Perform the task"
```

For ALOHA EEF conversion, `--eef-action-from-state-shift-frames=0` aligns an action target with
the current state when the raw episode does not provide `actions.eef_pose`. This option affects
data conversion, not normalization.

The converted directory should contain:

```text
canonical_tactile_task/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/<camera-key>/episode_000000.mp4
└── meta/
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── info.json
    └── tasks.jsonl
```

Confirm that `observation.state`, `action`, and `action_mask` are 32-dimensional, video keys
match `meta/info.json`, tactile streams are time-aligned, and every episode has a task entry.
See `python scripts/convert_canonical_data.py --help` for all conversion options.

### 2. Compute normalization statistics

Normalization statistics depend on the dataset, robot action layout, and delta-action
convention. Recompute them whenever any of these changes. Tactile images are not included in
this calculation.

```bash
export VTLA_DATASET_PATH=/path/to/datasets/canonical_tactile_task
export VTLA_ASSET_ID=canonical_tactile_task

python scripts/compute_canonical_norm.py \
  --repo-id "$VTLA_DATASET_PATH" \
  --robot flexiv \
  --train-config-name vtla_tactile_posttrain \
  --asset-id "$VTLA_ASSET_ID"
```

For ALOHA EEF data, use `--robot aloha`. The result is written to:

```text
assets/vtla_tactile_posttrain/<asset-id>/norm_stats.json
```

`--max-frames` can be used for a pipeline smoke test, but final statistics should be computed
over the full training set.

The normalization script rejects state/action columns that are not 32-dimensional numeric
vectors or contain NaN, infinity, or null values. List elements must have an integer or
floating-point Arrow type; numeric strings and booleans are rejected. The error identifies the Parquet file and
column; repair the source data before recomputing statistics. Validation failures leave any
existing normalization asset unchanged.

### 3. Configure post-training

The reference config reads local paths from environment variables:

```bash
export VTLA_DATASET_PATH=/path/to/datasets/canonical_tactile_task
export VTLA_PRETRAINED_CHECKPOINT=/path/to/checkpoints/vtla_pretrained
export VTLA_ASSET_ID=canonical_tactile_task
export VTLA_DEFAULT_PROMPT="Perform the task"
export EXP_NAME=my_experiment
export NPROC_PER_NODE=8
```

`VTLA_PRETRAINED_CHECKPOINT` must directly contain `model.safetensors`, and `VTLA_ASSET_ID` must
exactly match the asset id used to compute norm statistics.

The reference recipe uses:

| Setting | Value |
| --- | --- |
| action dimension / horizon | `32 / 50` |
| batch size / training steps | `64 / 20,000` |
| learning rate | 500-step warmup to `2e-5`, cosine decay to `2e-6` |
| gradient clipping | `1.0` |
| tactile inputs | four fixed slots with missing-view masks |
| VL dropout | `0.0` |
| supervised predictor loss | disabled |
| attention backend | eager |
| tactile-specific LR scale | `0.1` |

This is task-level post-training with the action objective. `predictor_loss_weight=0.0` disables the
separate supervised predictor objective, so the tactile branch is trained through the action loss
alone.

`VTLA_PREDICTOR_LR_SCALE=0.1` does not freeze the tactile branch. It updates tactile-specific
parameters at one tenth of the main learning rate through the action objective.

### 4. Validate and train

Run the preflight check before allocating a full job:

```bash
CHECK_ONLY=1 bash train.sh
```

It verifies GPU visibility, checkpoint and dataset paths, norm statistics, and the local
DINOv2 cache.

Start a new experiment:

```bash
export WANDB_MODE=offline
bash train.sh --overwrite
```

Resume the latest checkpoint of the same experiment:

```bash
bash train.sh --resume
```

Do not use `--overwrite` and `--resume` together.

The launcher defaults to eager attention because it matches the pretraining computation path:

```bash
VTLA_ATTN_IMPL=eager
VTLA_PREFIX_CACHE=1
VTLA_PREDICTOR_LR_SCALE=0.1
```

Treat SDPA as experimental unless it has been validated with the same checkpoint, precision,
masks, and training horizon.

Checkpoints are written to:

```text
checkpoints/vtla_tactile_posttrain/<experiment>/<step>/
├── model.safetensors
├── optimizer.pt
├── metadata.pt
└── assets/<asset-id>/norm_stats.json
```

For the first smoke test, verify that the action batch has shape `(batch, 50, 32)`, real tactile
views have true masks, missing views have false masks, all ranks load identical parameters, and
loss and gradient norm remain finite.

### Troubleshooting

- **`no episodes left after validation`.** Point `raw_root` at the directory whose immediate
  children are valid episode directories. Inspect the first rejected episode before moving or
  flattening the full dataset.
- **Missing norm statistics.** Confirm that `VTLA_ASSET_ID` matches `--asset-id` and that
  `assets/vtla_tactile_posttrain/$VTLA_ASSET_ID/norm_stats.json` exists.
- **Missing camera or tactile key.** Preserve the canonical slot and use a placeholder with a
  false mask. Do not replace it with another view.
- **Loss does not decrease or becomes non-finite.** Confirm eager attention, the intended
  pretrained checkpoint, fully loaded tactile-predictor parameters, current norm statistics, the
  correct action layout and delta mask, and aligned tactile baseline/current frames.
