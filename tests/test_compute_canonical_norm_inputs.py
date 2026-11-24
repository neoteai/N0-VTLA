"""Standalone normalization CLI regressions using temporary synthetic Parquet."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(os.environ.get("N0VTLA_ROOT", Path(__file__).resolve().parents[1]))
SCRIPT = ROOT / "scripts" / "compute_canonical_norm.py"
COLUMNS = ("observation.state", "action")


def write_episode(repo, index=0, robot="flexiv", **changes):
    state = np.zeros((3, 32), dtype=np.float32)
    action = np.zeros_like(state)
    active = 10 if robot == "flexiv" else 20
    for dim in range(active):
        state[:, dim] = [dim + 0.1, dim + 0.3, dim + 0.7]
        action[:, dim] = [dim + 0.2, dim + 0.6, dim + 1.1]
    state[:, 0] = [0, 1, 2]
    action[:, 0] = [1, 2, 3]
    state[:, 9] = [0.1, 0.2, 0.3]
    action[:, 9] = [0.2, 0.4, 0.6]
    if robot == "aloha":
        state[:, 10] = [10, 11, 12]
        action[:, 10] = [12, 13, 14]
        state[:, 19] = [0.3, 0.4, 0.5]
        action[:, 19] = [0.5, 0.7, 0.9]
    columns = {"observation.state": state.tolist(), "action": action.tolist()}
    columns.update(changes)
    path = repo / "data/chunk-000" / f"episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: pa.array(values) if name in changes else pa.array(values, type=pa.list_(pa.float32(), 32))
        for name, values in columns.items()
    }
    pq.write_table(pa.table(arrays), path)
    return path


def run_cli(repo, model, robot="flexiv"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-id", str(repo), "--robot", robot,
         "--train-config-name", "fixture", "--asset-id", "sample", "--repo-root", str(model)],
        capture_output=True, text=True, timeout=30,
    )


def output_path(model):
    return model / "assets/fixture/sample/norm_stats.json"


def assert_rejected(tmp_path, column, values, *, episode=0, existing=True):
    repo, model = tmp_path / "dataset", tmp_path / "model"
    if episode:
        write_episode(repo)
    bad_path = write_episode(repo, episode, **{column: values})
    original_inputs = {p: p.read_bytes() for p in repo.rglob("*.parquet")}
    target = output_path(model)
    sentinel = b'{"existing": "valid asset must survive"}\n'
    if existing:
        target.parent.mkdir(parents=True)
        target.write_bytes(sentinel)
    result = run_cli(repo, model)
    assert result.returncode != 0, result.stdout
    assert bad_path.name in result.stderr
    assert column in result.stderr
    if existing:
        assert target.read_bytes() == sentinel
    else:
        assert not target.exists()
        assert not model.exists()
    assert all(path.read_bytes() == data for path, data in original_inputs.items())
    return result


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("episode", [0, 1], ids=["first-episode", "later-episode"])
def test_nonfinite_column_preserves_existing_asset(tmp_path, column, bad, episode):
    values = np.zeros((3, 32)).tolist()
    values[1][0] = bad
    result = assert_rejected(tmp_path, column, values, episode=episode)
    assert "non-finite" in result.stderr.lower()


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("width", [31, 33])
def test_wrong_width_is_rejected(tmp_path, column, width):
    result = assert_rejected(tmp_path, column, np.zeros((3, width)).tolist())
    assert "32" in result.stderr


@pytest.mark.parametrize("column", COLUMNS)
def test_ragged_column_is_rejected_with_context(tmp_path, column):
    result = assert_rejected(tmp_path, column, [[0.0] * 32, [0.0] * 31, [0.0] * 32])
    assert "32" in result.stderr


@pytest.mark.parametrize("column", COLUMNS)
def test_null_element_is_rejected(tmp_path, column):
    values = np.zeros((3, 32)).tolist()
    values[1][5] = None
    assert_rejected(tmp_path, column, values)


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("value", ["0", False], ids=["numeric-string", "boolean"])
def test_nonnumeric_element_type_is_rejected(tmp_path, column, value):
    assert_rejected(tmp_path, column, [[value] * 32] * 3)


@pytest.mark.parametrize("column", COLUMNS)
def test_nonfinite_reserved_dimension_is_rejected(tmp_path, column):
    values = np.zeros((3, 32)).tolist()
    values[1][31] = float("nan")
    assert_rejected(tmp_path, column, values, episode=1)


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_column_reports_episode_and_preserves_asset(tmp_path, column):
    repo, model = tmp_path / "dataset", tmp_path / "model"
    path = write_episode(repo, index=123456)
    pq.write_table(pq.read_table(path).drop([column]), path)
    original = path.read_bytes()
    target = output_path(model)
    target.parent.mkdir(parents=True)
    sentinel = b'{"existing": "valid asset must survive"}\n'
    target.write_bytes(sentinel)
    result = run_cli(repo, model)
    assert result.returncode != 0
    assert path.name in result.stderr
    assert column in result.stderr
    assert target.read_bytes() == sentinel
    assert path.read_bytes() == original


def test_invalid_input_creates_no_output_asset(tmp_path):
    values = np.zeros((3, 32)).tolist()
    values[1][0] = float("nan")
    assert_rejected(tmp_path, "observation.state", values, existing=False)


@pytest.mark.parametrize("robot", ["flexiv", "aloha"])
def test_valid_eef_data_keeps_numeric_and_gripper_semantics(tmp_path, robot):
    repo, model = tmp_path / "dataset", tmp_path / "model"
    path = write_episode(repo, robot=robot)
    result = run_cli(repo, model, robot)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path(model).read_text())
    stats = payload["norm_stats"]
    table = pq.read_table(path)
    assert table.schema.field("action").type == pa.list_(pa.float32(), 32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    active = 10 if robot == "flexiv" else 20
    # Enumerate each reference vector independently of the script's helpers.
    expected_actions = []
    for frame in range(3):
        for step in range(50):
            vector = action[min(frame + step, 2)].copy()
            for dim in range(active):
                if dim not in (9, 19):
                    vector[dim] -= state[frame, dim]
            expected_actions.append(vector)
    for name, expected in (("state", state), ("actions", expected_actions)):
        expected = np.asarray(expected, dtype=np.float64)
        np.testing.assert_allclose(stats[name]["mean"][:active], expected.mean(axis=0)[:active], atol=1e-6)
        np.testing.assert_allclose(stats[name]["std"][:active], expected.std(axis=0)[:active], atol=1e-6)
    # Three frames, horizon 50: left x delta sum = 147 + 99 + 50.
    assert stats["state"]["mean"][0] == pytest.approx(1.0)
    assert stats["actions"]["mean"][0] == pytest.approx(296 / 150)
    # Absolute left gripper sum = 29.4 + 29.8 + 30.0, without state subtraction.
    assert stats["actions"]["mean"][9] == pytest.approx(89.2 / 150, rel=1e-6)
    if robot == "aloha":
        assert stats["actions"]["mean"][10] == pytest.approx(446 / 150)
        assert stats["actions"]["mean"][19] == pytest.approx(134.2 / 150, rel=1e-6)
    for group in stats.values():
        for field, values in group.items():
            assert len(values) == 32
            assert np.isfinite(values).all()
            identity = {"mean": 0, "std": 1, "q01": -1, "q99": 1}[field]
            np.testing.assert_array_equal(values[active:], identity)
