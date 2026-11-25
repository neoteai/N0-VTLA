"""CLI regressions for N0-VTLA raw/output directory overlap.

Run with: N0VTLA_ROOT=/path/to/N0-VTLA pytest -q test_convert_canonical_paths.py
Only temporary synthetic fixtures are used; no robot or checkpoints are needed.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(os.environ.get('N0VTLA_ROOT', Path(__file__).resolve().parents[1])).resolve() / 'scripts/convert_canonical_data.py'
COLS = ('x', 'y', 'z', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'gripper')
VIDEO_KEYS = (
    'observation.image.third_view',
    'observation.image.left_wrist_view',
    'observation.image.left_wrist_left_tactile',
    'observation.image.left_wrist_right_tactile',
)


def make_episode(raw):
    episode = raw / 'episode_000000'
    episode.mkdir(parents=True)
    (episode / 'metadata.json').write_text(json.dumps({'total_frames': 3}))
    for folder in ('observation.state.eef_pose', 'actions.eef_pose'):
        path = episode / folder / 'data.csv'
        path.parent.mkdir()
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLS)
            for frame in range(3):
                writer.writerow((frame, 0, 0, 1, 0, 0, 0, 1, 0, 0.5))
    # Deliberately opaque bytes: these tests exercise conversion/path protection,
    # not decoding, synchronization, or validity of tactile video.
    for key in VIDEO_KEYS:
        folder = episode / key
        folder.mkdir()
        (folder / 'video.mp4').write_bytes(b'synthetic-copy-fixture-not-a-real-video')
    return episode


def snapshot(root):
    entries = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            value = ('symlink', os.readlink(path))
        elif path.is_dir():
            value = ('directory',)
        else:
            value = ('file', path.read_bytes())
        entries[str(path.relative_to(root))] = value
    return entries


def run_conversion(raw, output, overwrite=False):
    command = [sys.executable, str(SCRIPT), str(raw), str(output), '--robot', 'flexiv']
    if overwrite:
        command.append('--overwrite')
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('relation', [
    'same', 'ancestor', 'descendant', 'new_descendant',
    'symlink_same', 'symlink_ancestor',
    'case_same', 'case_ancestor', 'case_new_descendant',
])
@pytest.mark.parametrize('overwrite', [False, True])
def test_overlapping_directories_preserve_every_source_byte(tmp_path, relation, overwrite):
    raw = tmp_path / 'Dataset' / 'RawCase'
    make_episode(raw)
    if relation == 'same':
        output = raw
    elif relation == 'ancestor':
        output = raw.parent
    elif relation == 'descendant':
        output = raw / 'episode_000000'
    elif relation == 'new_descendant':
        output = raw / 'not-yet-existing'
    elif relation.startswith('case_'):
        target = raw.parent if relation == 'case_ancestor' else raw
        alias = target.with_name(target.name.swapcase())
        if not alias.exists() or not alias.samefile(target):
            pytest.skip('requires case-insensitive path lookup')
        output = alias / 'not-yet-existing' if relation == 'case_new_descendant' else alias
    else:
        output = tmp_path / 'alias'
        output.symlink_to(raw if relation == 'symlink_same' else raw.parent, target_is_directory=True)
    before = snapshot(raw)
    result = run_conversion(raw, output, overwrite)
    assert snapshot(raw) == before
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'must not overlap' in result.stderr
    if relation in ('new_descendant', 'case_new_descendant'):
        assert not output.exists()


def test_sibling_conversion_produces_parquet_and_preserves_source(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    make_episode(raw)
    before = snapshot(raw)
    result = run_conversion(raw, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(raw) == before
    table = pq.read_table(output / 'data/chunk-000/episode_000000.parquet')
    assert table.num_rows == 3
    assert len(table['action'][0].as_py()) == 32
    assert table['action_mask'][0].as_py() == [True] * 10 + [False] * 22
    state = np.asarray(table['observation.state'].to_pylist())
    assert state[:, 0].tolist() == [0, 1, 2]
    assert np.count_nonzero(state[:, 10:]) == 0
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['total_episodes'] == 1 and info['total_frames'] == 3
    assert all((output / f'videos/chunk-000/{key}/episode_000000.mp4').is_file() for key in VIDEO_KEYS)


def test_sibling_overwrite_replaces_only_output(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    make_episode(raw)
    before = snapshot(raw)
    output.mkdir()
    sentinel = output / 'old-output.txt'
    sentinel.write_text('old converted data')
    result = run_conversion(raw, output, overwrite=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()
    assert snapshot(raw) == before


def test_existing_sibling_requires_overwrite(tmp_path):
    raw, output = tmp_path / 'raw', tmp_path / 'converted'
    make_episode(raw)
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep')
    before = snapshot(raw)
    result = run_conversion(raw, output)
    assert result.returncode != 0
    assert 'FileExistsError' in result.stderr
    assert sentinel.read_text() == 'keep'
    assert snapshot(raw) == before
