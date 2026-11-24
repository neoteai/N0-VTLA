#!/usr/bin/env python3
"""Compute training normalization statistics for a canonical LeRobot dataset.

The script reads 32-dimensional ``observation.state`` and ``action`` columns,
constructs a 50-step action horizon, applies the selected robot's element-wise
delta convention, and writes an OpenPI-compatible ``norm_stats.json``.

Example:
  python scripts/compute_canonical_norm.py \
    --repo-id /path/to/dataset \
    --robot flexiv \
    --train-config-name my_flexiv_predictor \
    --asset-id my_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_HORIZON = 50
NUM_QUANTILE_BINS = 5000

# A dimension whose observed spread is below this is treated as constant (reserved padding or an
# unused arm block) and gets identity normalisation stats. See RunningStats.as_json.
_CONSTANT_DIM_TOL = 1e-8

ROBOT_DELTA_MASKS = {
    "flexiv": [True] * 9 + [False],
    "aloha": [True] * 9 + [False] + [True] * 9 + [False],
}


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = None
        self.mean_of_squares = None
        self.min = None
        self.max = None
        self.histograms = None
        self.bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        if batch.shape[0] == 0:
            return
        num_elements, vector_length = batch.shape
        if self.count == 0:
            self.mean = np.mean(batch, axis=0)
            self.mean_of_squares = np.mean(batch**2, axis=0)
            self.min = np.min(batch, axis=0)
            self.max = np.max(batch, axis=0)
            self.histograms = [np.zeros(NUM_QUANTILE_BINS, dtype=np.float64) for _ in range(vector_length)]
            self.bin_edges = [
                np.linspace(self.min[i] - 1e-10, self.max[i] + 1e-10, NUM_QUANTILE_BINS + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self.mean.size:
                raise ValueError("Vector length changed.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self.max)
            min_changed = np.any(new_min < self.min)
            self.max = np.maximum(self.max, new_max)
            self.min = np.minimum(self.min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self.count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (batch_mean_of_squares - self.mean_of_squares) * (num_elements / self.count)
        self._update_histograms(batch)

    def _adjust_histograms(self) -> None:
        for i in range(len(self.histograms)):
            old_edges = self.bin_edges[i]
            new_edges = np.linspace(self.min[i], self.max[i], NUM_QUANTILE_BINS + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self.histograms[i])
            self.histograms[i] = new_hist
            self.bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self.bin_edges[i])
            self.histograms[i] += hist

    def _quantile(self, q: float) -> np.ndarray:
        target_count = q * self.count
        values = []
        for hist, edges in zip(self.histograms, self.bin_edges, strict=True):
            cumsum = np.cumsum(hist)
            idx = min(np.searchsorted(cumsum, target_count), len(edges) - 1)
            values.append(edges[idx])
        return np.asarray(values)

    def as_json(self) -> dict[str, list[float]]:
        if self.count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self.mean_of_squares - self.mean**2
        std = np.sqrt(np.maximum(0, variance))
        q01 = self._quantile(0.01)
        q99 = self._quantile(0.99)

        # Identity stats for constant dimensions.
        #
        # The canonical 32-dim container reserves dims 20:32, and a single-arm dataset also
        # leaves the whole second-arm block constant at zero. Statistics measured over a
        # constant-zero dimension come out as q01 == q99 == 0 and std == 0, and quantile
        # normalisation then maps the constant 0 to -1 rather than to 0. The pretrained
        # checkpoints normalise those same dimensions with identity stats, so 0 maps to 0.
        #
        # Mixing the two conventions is not fatal -- training pulls the padded outputs back and
        # converges to the same place -- but it starts post-training with a step-0 loss around
        # 2.5 instead of 0.04, which reads as a broken run. Emit identity stats for constant
        # dimensions so a freshly computed asset agrees with the released weights.
        constant = (q99 - q01 <= _CONSTANT_DIM_TOL) & (std <= _CONSTANT_DIM_TOL)
        mean = np.where(constant, 0.0, self.mean)
        std = np.where(constant, 1.0, std)
        q01 = np.where(constant, -1.0, q01)
        q99 = np.where(constant, 1.0, q99)
        if constant.any():
            logging.info(
                "identity stats applied to %d constant dimension(s): %s",
                int(constant.sum()),
                np.flatnonzero(constant).tolist(),
            )

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def read_fixed_list(table, name: str) -> np.ndarray:
    column_type = table.schema.field(name).type
    if not (
        pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
        or pa.types.is_fixed_size_list(column_type)
    ) or not (
        pa.types.is_integer(column_type.value_type) or pa.types.is_floating(column_type.value_type)
    ):
        raise ValueError(f"{name}: expected a list of integer or floating-point values, got {column_type}")
    try:
        values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected rows of 32 numeric values") from exc
    if values.ndim != 2 or values.shape[1] != 32:
        raise ValueError(f"{name}: expected shape (frames, 32), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains non-finite values (NaN, infinity, or null)")
    return values


def action_horizon_sequence(actions: np.ndarray, horizon: int = ACTION_HORIZON) -> np.ndarray:
    frame_count = actions.shape[0]
    offsets = np.arange(horizon, dtype=np.int64)
    base = np.arange(frame_count, dtype=np.int64)[:, None]
    indices = np.minimum(base + offsets[None, :], frame_count - 1)
    return actions[indices]


def apply_delta(actions: np.ndarray, state: np.ndarray, delta_mask: list[bool]) -> np.ndarray:
    actions = actions.copy()
    mask = np.asarray(delta_mask, dtype=bool)
    dims = mask.shape[0]
    subtract = np.where(mask, state[:, :dims], 0.0).astype(actions.dtype)
    actions[..., :dims] -= subtract[:, None, :]
    return actions


def compute(spec: dict, max_frames: int | None) -> tuple[dict, int]:
    repo = Path(spec["repo_id"])
    if not (repo / "data").is_dir():
        raise FileNotFoundError(f"dataset data directory not found: {repo / 'data'}")
    state_stats = RunningStats()
    action_stats = RunningStats()
    processed = 0

    for parquet_path in sorted((repo / "data").rglob("*.parquet")):
        try:
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            state = read_fixed_list(table, "observation.state")
            action = read_fixed_list(table, "action")
        except (pa.ArrowException, ValueError) as exc:
            raise ValueError(f"{parquet_path}: {exc}") from exc

        if max_frames is not None:
            remaining = max_frames - processed
            if remaining <= 0:
                break
            state = state[:remaining]
            action = action[:remaining]

        actions = action_horizon_sequence(action)
        actions = apply_delta(actions, state, spec["delta_mask"])
        state_stats.update(state)
        action_stats.update(actions)
        processed += state.shape[0]

        if processed % 50000 < state.shape[0]:
            print(f"processed {processed} frames")
        if max_frames is not None and processed >= max_frames:
            break

    return {"norm_stats": {"state": state_stats.as_json(), "actions": action_stats.as_json()}}, processed


def resolve_spec(args: argparse.Namespace) -> tuple[str, dict]:
    repo = Path(args.repo_id)
    return args.train_config_name, {
        "repo_id": str(repo),
        "asset_id": args.asset_id or repo.name,
        "delta_mask": ROBOT_DELTA_MASKS[args.robot],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Local canonical LeRobot dataset root.")
    parser.add_argument(
        "--robot",
        required=True,
        choices=sorted(ROBOT_DELTA_MASKS),
        help="Robot layout used to select the element-wise delta mask.",
    )
    parser.add_argument(
        "--train-config-name",
        required=True,
        help="TrainConfig.name used to choose assets/<train-config-name>/<asset-id>/norm_stats.json.",
    )
    parser.add_argument("--asset-id", help="Asset id directory. Defaults to basename of --repo-id.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for the assets directory. Defaults to the parent of scripts/.",
    )
    args = parser.parse_args()

    train_config_name, spec = resolve_spec(args)
    payload, processed = compute(spec, args.max_frames)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = repo_root / "assets" / train_config_name / spec["asset_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Processed frames: {processed}")
    print(f"Writing stats to: {output_path}")


if __name__ == "__main__":
    main()
