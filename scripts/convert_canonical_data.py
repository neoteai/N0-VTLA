#!/usr/bin/env python3
r"""Convert raw robot episodes into a canonical LeRobot dataset.

Examples:
  python scripts/convert_canonical_data.py \
    /path/to/raw-flexiv-data \
    /path/to/converted-flexiv-data \
    --robot flexiv --task "do the task"

  python scripts/convert_canonical_data.py \
    /path/to/raw-aloha-data \
    /path/to/converted-aloha-data \
    --robot aloha --task "do the task"

  # Aloha defaults to bimanual EEF rot6d. To use joint-space conversion:
  #   --robot aloha --aloha-action-space joint_position

The output uses the per-episode LeRobot v2.1 layout:

  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<video_key>/episode_000000.mp4
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/info.json

The parquet rows contain canonical 32-dim ``observation.state`` / ``action`` /
``action_mask`` plus frame metadata. Videos are referenced through
``meta/info.json`` and are not stored as parquet columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_FPS = 30.0
CANONICAL_ACTION_DIM = 32
CHUNK_SIZE = 1000

FLEXIV_EEF_COLS = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper")
FLEXIV_EEF_ACTION_ALIASES = (
    ("x", "tcp.x"),
    ("y", "tcp.y"),
    ("z", "tcp.z"),
    ("r1", "tcp.r1"),
    ("r2", "tcp.r2"),
    ("r3", "tcp.r3"),
    ("r4", "tcp.r4"),
    ("r5", "tcp.r5"),
    ("r6", "tcp.r6"),
    ("gripper", "gripper.pos"),
)
ALOHA_JOINT_COLS = (
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
)
ALOHA_EEF_COLS = (
    "left_x", "left_y", "left_z",
    "left_r1", "left_r2", "left_r3", "left_r4", "left_r5", "left_r6",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_r1", "right_r2", "right_r3", "right_r4", "right_r5", "right_r6",
    "right_gripper",
)

FLEXIV_VIDEO_KEY_ALIASES = {
    "observation.image.third_view": ("observation.image.third_view",),
    "observation.image.left_wrist_view": (
        "observation.image.left_wrist_view",
        "observation.image.right_wrist_view",
    ),
    "observation.image.left_wrist_left_tactile": ("observation.image.left_wrist_left_tactile",),
    "observation.image.left_wrist_right_tactile": ("observation.image.left_wrist_right_tactile",),
}

ALOHA_VIDEO_KEY_ALIASES = {
    "observation.image.third_view": ("observation.image.third_view",),
    "observation.image.left_wrist_view": ("observation.image.left_wrist_view",),
    "observation.image.right_wrist_view": ("observation.image.right_wrist_view",),
    "observation.image.left_wrist_left_tactile": ("observation.image.left_wrist_left_tactile",),
    "observation.image.left_wrist_right_tactile": ("observation.image.left_wrist_right_tactile",),
    "observation.image.right_wrist_left_tactile": ("observation.image.right_wrist_left_tactile",),
    "observation.image.right_wrist_right_tactile": ("observation.image.right_wrist_right_tactile",),
}


def _robot_profile(robot: str, *, aloha_action_space: str = "eef") -> dict[str, Any]:
    if robot == "flexiv":
        return {
            "robot_type": "single_arm_tactile",
            "video_key_aliases": FLEXIV_VIDEO_KEY_ALIASES,
            "state_csv": "observation.state.eef_pose/data.csv",
            "state_cols": FLEXIV_EEF_COLS,
            "action_csv": "actions.eef_pose/data.csv",
            "action_aliases": FLEXIV_EEF_ACTION_ALIASES,
        }
    if robot == "aloha":
        if aloha_action_space == "eef":
            return {
                "robot_type": "bimanual_tactile_eef",
                "video_key_aliases": ALOHA_VIDEO_KEY_ALIASES,
                "state_csv": "observation.state.eef_pose/data.csv",
                "state_cols": ALOHA_EEF_COLS,
                "action_csv": "actions.eef_pose/data.csv",
                "action_aliases": tuple((col,) for col in ALOHA_EEF_COLS),
            }
        if aloha_action_space != "joint_position":
            raise ValueError(f"unsupported Aloha action space: {aloha_action_space}")
        return {
            "robot_type": "bimanual_tactile",
            "video_key_aliases": ALOHA_VIDEO_KEY_ALIASES,
            "state_csv": "observation.state.joint_position/data.csv",
            "state_cols": ALOHA_JOINT_COLS,
            "action_csv": "actions.joint_position/data.csv",
            "action_aliases": tuple((col,) for col in ALOHA_JOINT_COLS),
        }
    raise ValueError(f"unsupported robot: {robot}")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_float_alias(row: dict[str, str], aliases: tuple[str, ...]) -> float:
    for alias in aliases:
        if alias in row:
            return float(row[alias])
    raise KeyError(f"missing any of columns {aliases}; available columns={tuple(row)}")


def _array_from_cols(rows: list[dict[str, str]], cols: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float32)


def _array_from_aliases(
    rows: list[dict[str, str]],
    aliases: tuple[tuple[str, ...], ...],
    *,
    gripper_scale: float | None = None,
) -> np.ndarray:
    values = np.asarray(
        [[_read_float_alias(row, col_aliases) for col_aliases in aliases] for row in rows],
        dtype=np.float32,
    )
    if gripper_scale is not None:
        values[:, -1] *= np.float32(gripper_scale)
    return values


def _paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    if first.is_relative_to(second) or second.is_relative_to(first):
        return True
    # resolve() does not canonicalize case on case-insensitive filesystems.
    # Compare existing ancestors by identity, including for a new output path.
    for root, other in ((first, second), (second, first)):
        if not root.exists():
            continue
        for ancestor in (other, *other.parents):
            if ancestor.exists() and root.samefile(ancestor):
                return True
    return False


def _reset_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _parse_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantile_stats(values: np.ndarray) -> dict[str, list[float | int | bool]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        raise ValueError("cannot compute stats for empty array")

    if arr.dtype == np.bool_:
        arr64 = arr.astype(np.float64)
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr64.mean(axis=0).tolist(),
            "std": arr64.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }

    arr64 = arr.astype(np.float64, copy=False)
    return {
        "min": arr64.min(axis=0).tolist(),
        "max": arr64.max(axis=0).tolist(),
        "mean": arr64.mean(axis=0).tolist(),
        "std": arr64.std(axis=0).tolist(),
        "count": [int(arr64.shape[0])],
    }


def _fixed_size_list_array(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1).tolist(), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _pad_to_width(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape={values.shape}")
    if values.shape[1] > width:
        raise ValueError(f"cannot pad width {values.shape[1]} to smaller width {width}")
    mask = np.zeros((values.shape[0], width), dtype=bool)
    mask[:, : values.shape[1]] = True
    if values.shape[1] == width:
        return values, mask
    padded = np.zeros((values.shape[0], width), dtype=np.float32)
    padded[:, : values.shape[1]] = values
    return padded, mask


def _hf_schema_metadata() -> dict[bytes, bytes]:
    features = {
        "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "action_mask": {"feature": {"dtype": "bool", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def _copy_video(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _resolve_video_path(episode_dir: Path, video_key_aliases: dict[str, tuple[str, ...]], video_key: str) -> Path | None:
    for candidate_key in video_key_aliases[video_key]:
        path = episode_dir / candidate_key / "video.mp4"
        if path.exists():
            return path
    return None


def _video_feature() -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(DATASET_FPS),
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _build_info_json(
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    data_size_bytes: int,
    video_size_bytes: int,
    video_keys: tuple[str, ...],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [32], "names": None},
        "action": {"dtype": "float32", "shape": [32], "names": None},
        "action_mask": {"dtype": "bool", "shape": [32], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in video_keys:
        features[key] = _video_feature()

    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": math.ceil(data_size_bytes / (1024 * 1024)),
        "video_files_size_in_mb": math.ceil(video_size_bytes / (1024 * 1024)),
        "fps": DATASET_FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def _collect_episode_dirs(raw_root: Path) -> list[Path]:
    candidates = [
        *raw_root.glob("episode_*"),
        *raw_root.glob("2026*/**/episode_*"),
    ]
    return sorted(dict.fromkeys(path for path in candidates if path.is_dir()))


def _episode_drop_reasons(
    episode_dir: Path,
    *,
    profile: dict[str, Any],
    require_action_csv: bool = True,
) -> list[str]:
    reasons: list[str] = []
    required_paths = [Path("metadata.json"), Path(profile["state_csv"])]
    if require_action_csv:
        required_paths.append(Path(profile["action_csv"]))
    for rel_path in required_paths:
        if not (episode_dir / rel_path).exists():
            reasons.append(f"missing_{rel_path.as_posix()}")

    video_key_aliases = profile["video_key_aliases"]
    for video_key in video_key_aliases:
        if _resolve_video_path(episode_dir, video_key_aliases, video_key) is None:
            reasons.append(f"missing_video={video_key}")

    if reasons:
        return reasons

    metadata = _parse_metadata(episode_dir / "metadata.json")
    expected_rows = int(metadata["total_frames"])
    csv_specs = [("observation.state", episode_dir / profile["state_csv"])]
    if require_action_csv:
        csv_specs.insert(0, ("action", episode_dir / profile["action_csv"]))
    csv_row_counts: dict[str, int] = {}
    for name, path in csv_specs:
        rows = _read_csv_dicts(path)
        csv_row_counts[name] = len(rows)
        if len(rows) <= 0:
            reasons.append(f"empty_csv_{name}")
        if len(rows) > expected_rows or expected_rows - len(rows) > 1:
            reasons.append(f"row_mismatch_{name}={len(rows)}_expected_{expected_rows}")
    if len(set(csv_row_counts.values())) > 1:
        reasons.append(f"csv_row_count_disagreement={csv_row_counts}")
    return reasons


def _load_excluded_episode_names(path: Path | None) -> set[str]:
    if path is None:
        return set()

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            items = (
                payload.get("exclude_reasons")
                or payload.get("exclude_episodes")
                or payload.get("excluded_episodes")
                or payload.get("episodes")
            )
            if isinstance(items, dict):
                return {str(key) for key in items}
            if isinstance(items, list):
                return {str(item) for item in items}
        if isinstance(payload, list):
            return {str(item) for item in payload}
        raise ValueError(f"unsupported exclude JSON format: {path}")

    return {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    action_mask: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: int,
    index: np.ndarray,
) -> int:
    length = state.shape[0]
    table = pa.Table.from_arrays(
        [
            _fixed_size_list_array(state, 32, pa.float32()),
            _fixed_size_list_array(action, 32, pa.float32()),
            _fixed_size_list_array(action_mask, 32, pa.bool_()),
            pa.array(timestamp.tolist(), type=pa.float32()),
            pa.array(frame_index.tolist(), type=pa.int64()),
            pa.array([episode_index] * length, type=pa.int64()),
            pa.array(index.tolist(), type=pa.int64()),
            pa.array([0] * length, type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "action_mask",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).replace_schema_metadata(_hf_schema_metadata())
    pq.write_table(table, path, compression="zstd")
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_root", type=Path, help="Root directory containing raw episode folders.")
    parser.add_argument("output_dir", type=Path, help="Output local LeRobot dataset root.")
    parser.add_argument("--robot", choices=("flexiv", "aloha"), required=True, help="Raw robot schema to convert.")
    parser.add_argument(
        "--aloha-action-space",
        choices=("eef", "joint_position"),
        default="eef",
        help="Aloha only: convert canonical state/action as bimanual EEF rot6d or legacy joint_position.",
    )
    parser.add_argument("--task", default="do the task", help="Task string written to meta/tasks.jsonl.")
    parser.add_argument(
        "--eef-action-from-state-shift-frames",
        type=int,
        default=None,
        help=(
            "EEF action-space only: derive absolute EEF action targets from future state frames. "
            "Aloha EEF defaults to 0 because raw Aloha episodes usually do not contain actions.eef_pose."
        ),
    )
    parser.add_argument("--eef-action-gripper-scale", type=float, default=None, help="Flexiv only: scale gripper action.")
    parser.add_argument("--exclude-episodes-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    args = parser.parse_args()

    if _paths_overlap(args.raw_root, args.output_dir):
        parser.error("raw_root and output_dir must not overlap; choose separate dataset directories")

    if args.robot != "aloha" and args.aloha_action_space != "eef":
        parser.error("--aloha-action-space is only supported for --robot aloha")
    action_space_is_eef = args.robot == "flexiv" or (args.robot == "aloha" and args.aloha_action_space == "eef")
    if not action_space_is_eef and args.eef_action_from_state_shift_frames is not None:
        parser.error("--eef-action-from-state-shift-frames is only supported for EEF action-space")
    if args.robot != "flexiv" and args.eef_action_gripper_scale is not None:
        parser.error("--eef-action-gripper-scale is only supported for --robot flexiv")
    if args.eef_action_from_state_shift_frames is not None and args.eef_action_from_state_shift_frames < 0:
        parser.error("--eef-action-from-state-shift-frames must be non-negative")
    if args.robot == "aloha" and args.aloha_action_space == "eef" and args.eef_action_from_state_shift_frames is None:
        args.eef_action_from_state_shift_frames = 0

    profile = _robot_profile(args.robot, aloha_action_space=args.aloha_action_space)
    require_action_csv = args.eef_action_from_state_shift_frames is None
    video_key_aliases: dict[str, tuple[str, ...]] = profile["video_key_aliases"]
    video_keys = tuple(video_key_aliases)

    episode_dirs = _collect_episode_dirs(args.raw_root)
    if not episode_dirs:
        raise SystemExit(f"no episode directories found under {args.raw_root}")
    excluded_episode_names = _load_excluded_episode_names(args.exclude_episodes_file)

    _reset_output_dir(args.output_dir, overwrite=args.overwrite)
    (args.output_dir / "meta").mkdir(parents=True, exist_ok=True)
    for chunk_idx in range(math.ceil(max(1, len(episode_dirs)) / CHUNK_SIZE)):
        (args.output_dir / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        for video_key in video_keys:
            (args.output_dir / "videos" / f"chunk-{chunk_idx:03d}" / video_key).mkdir(parents=True, exist_ok=True)

    dropped: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    source_episode_map: list[dict[str, Any]] = []
    state_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    action_mask_all: list[np.ndarray] = []
    timestamp_all: list[np.ndarray] = []
    frame_index_all: list[np.ndarray] = []
    episode_index_all: list[np.ndarray] = []
    index_all: list[np.ndarray] = []

    current_global_index = 0
    data_size_bytes = 0
    video_size_bytes = 0
    raw_action_dim: int | None = None

    for source_episode_dir in episode_dirs:
        source_episode_name = source_episode_dir.name
        metadata_path = source_episode_dir / "metadata.json"
        metadata = _parse_metadata(metadata_path) if metadata_path.exists() else {}
        source_episode_id = str(metadata.get("episode_id", source_episode_name))
        if source_episode_name in excluded_episode_names or source_episode_id in excluded_episode_names:
            dropped.append(
                {"source_episode": source_episode_name, "episode_id": source_episode_id, "reasons": ["excluded_by_file"]}
            )
            continue

        reasons = _episode_drop_reasons(source_episode_dir, profile=profile, require_action_csv=require_action_csv)
        if reasons:
            dropped.append({"source_episode": source_episode_name, "episode_id": source_episode_id, "reasons": reasons})
            continue

        state_rows = _read_csv_dicts(source_episode_dir / profile["state_csv"])
        action_rows = _read_csv_dicts(source_episode_dir / profile["action_csv"]) if require_action_csv else []
        length = len(state_rows)
        episode_index = len(kept)
        episode_chunk = episode_index // CHUNK_SIZE
        frame_index = np.arange(length, dtype=np.int64)
        timestamp = (frame_index.astype(np.float32) / np.float32(DATASET_FPS)).astype(np.float32, copy=False)
        dataset_index = np.arange(current_global_index, current_global_index + length, dtype=np.int64)

        state_raw = _array_from_cols(state_rows, profile["state_cols"])
        if args.eef_action_from_state_shift_frames is not None:
            shifted_indices = np.minimum(
                np.arange(length, dtype=np.int64) + args.eef_action_from_state_shift_frames,
                length - 1,
            )
            action_raw = state_raw[shifted_indices].astype(np.float32, copy=True)
        else:
            action_raw = _array_from_aliases(
                action_rows,
                profile["action_aliases"],
                gripper_scale=args.eef_action_gripper_scale if args.robot == "flexiv" else None,
            )

        raw_action_dim = int(action_raw.shape[1])
        state, _ = _pad_to_width(state_raw, CANONICAL_ACTION_DIM)
        action, action_mask = _pad_to_width(action_raw, CANONICAL_ACTION_DIM)

        parquet_path = args.output_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        data_size_bytes += _write_episode_parquet(
            parquet_path,
            state=state,
            action=action,
            action_mask=action_mask,
            timestamp=timestamp,
            frame_index=frame_index,
            episode_index=episode_index,
            index=dataset_index,
        )

        for video_key in video_keys:
            src_video = _resolve_video_path(source_episode_dir, video_key_aliases, video_key)
            if src_video is None:
                raise FileNotFoundError(f"missing video for {video_key} in {source_episode_dir}")
            dst_video = (
                args.output_dir
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            _copy_video(src_video, dst_video)
            video_size_bytes += dst_video.stat().st_size

        episodes_rows.append({"episode_index": episode_index, "tasks": [args.task], "length": length})
        episodes_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "observation.state": _quantile_stats(state),
                    "action": _quantile_stats(action),
                    "action_mask": _quantile_stats(action_mask),
                    "timestamp": _quantile_stats(timestamp),
                    "frame_index": _quantile_stats(frame_index),
                    "episode_index": _quantile_stats(np.full(length, episode_index, dtype=np.int64)),
                    "index": _quantile_stats(dataset_index),
                    "task_index": _quantile_stats(np.zeros(length, dtype=np.int64)),
                },
            }
        )
        source_episode_map.append(
            {
                "output_episode_index": episode_index,
                "source_episode": source_episode_name,
                "source_episode_id": source_episode_id,
                "length": length,
            }
        )
        kept.append(
            {
                "source_episode": source_episode_name,
                "source_episode_id": source_episode_id,
                "output_episode_index": episode_index,
                "length": length,
            }
        )

        state_all.append(state)
        action_all.append(action)
        action_mask_all.append(action_mask)
        timestamp_all.append(timestamp)
        frame_index_all.append(frame_index)
        episode_index_all.append(np.full(length, episode_index, dtype=np.int64))
        index_all.append(dataset_index)
        current_global_index += length

    if not kept:
        raise SystemExit("no episodes left after validation")

    state_full = np.concatenate(state_all, axis=0)
    action_full = np.concatenate(action_all, axis=0)
    action_mask_full = np.concatenate(action_mask_all, axis=0)
    timestamp_full = np.concatenate(timestamp_all, axis=0)
    frame_index_full = np.concatenate(frame_index_all, axis=0)
    episode_index_full = np.concatenate(episode_index_all, axis=0)
    index_full = np.concatenate(index_all, axis=0)
    task_index_full = np.zeros(index_full.shape[0], dtype=np.int64)

    _write_jsonl(args.output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
    _write_jsonl(args.output_dir / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(args.output_dir / "meta" / "episodes_stats.jsonl", episodes_stats_rows)

    norm_stats = {
        "observation.state": _quantile_stats(state_full),
        "action": _quantile_stats(action_full),
        "action_mask": _quantile_stats(action_mask_full),
        "timestamp": _quantile_stats(timestamp_full),
        "frame_index": _quantile_stats(frame_index_full),
        "episode_index": _quantile_stats(episode_index_full),
        "index": _quantile_stats(index_full),
        "task_index": _quantile_stats(task_index_full),
    }
    (args.output_dir / "meta" / "norm_stats.json").write_text(json.dumps(norm_stats, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "meta" / "norm_stats_summary.json").write_text(
        json.dumps({"keys": sorted(norm_stats), "total_frames": int(index_full.shape[0])}, ensure_ascii=False, indent=2) + "\n"
    )

    info_payload = _build_info_json(
        robot_type=profile["robot_type"],
        total_episodes=len(kept),
        total_frames=int(index_full.shape[0]),
        data_size_bytes=data_size_bytes,
        video_size_bytes=video_size_bytes,
        video_keys=video_keys,
    )
    (args.output_dir / "meta" / "info.json").write_text(json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n")

    root_metadata = {
        "metadata_version": 1,
        "robot_type": profile["robot_type"],
        "fps": DATASET_FPS,
        "canonical_schema": True,
        "state_source_key": profile["state_csv"],
        "action_source_key": profile["state_csv"] if args.eef_action_from_state_shift_frames is not None else profile["action_csv"],
        "raw_action_dim": raw_action_dim,
        "canonical_action_dim": CANONICAL_ACTION_DIM,
        "total_episodes": len(kept),
        "total_frames": int(index_full.shape[0]),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(root_metadata, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "meta" / "source_episode_map.json").write_text(
        json.dumps(source_episode_map, ensure_ascii=False, indent=2) + "\n"
    )

    report = {
        "robot": args.robot,
        "task": args.task,
        "state_source_key": root_metadata["state_source_key"],
        "action_source_key": root_metadata["action_source_key"],
        "raw_action_dim": raw_action_dim,
        "canonical_action_dim": CANONICAL_ACTION_DIM,
        "kept_episode_count": len(kept),
        "dropped_episode_count": len(dropped),
        "kept_episodes": kept,
        "dropped_episodes": dropped,
    }
    (args.output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"output_dir: {args.output_dir}")
    print(f"robot: {args.robot}")
    print(f"kept episodes: {len(kept)}")
    print(f"dropped episodes: {len(dropped)}")
    print(f"total frames: {int(index_full.shape[0])}")
    print("layout: canonical per-episode parquet/jsonl")
    print(f"state width: {state_full.shape[1]}")
    print(f"action width: {action_full.shape[1]}")
    print(f"raw action width: {raw_action_dim}")


if __name__ == "__main__":
    main()

