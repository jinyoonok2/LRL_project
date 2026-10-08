from pathlib import Path

import pytest

from lrl_molmospaces_extensions.dataset.split import HOUSE, TRAJECTORY, OpenedWithGraspSplit, SplitClip


def _clips() -> list[SplitClip]:
    return [
        SplitClip("0000", "train", "house_1", "000001", "push"),
        SplitClip("0001", "train", "house_1", "000002", "push"),
        SplitClip("0002", "train", "house_2", "000001", "pull"),
        SplitClip("0003", "val", "house_3", "000001", "push"),
        SplitClip("0004", "val", "house_3", "000002", "pull"),
        SplitClip("0005", "train", "house_4", "000001", "pull"),
        SplitClip("0006", "train", "house_2", "000002", "pull"),
        SplitClip("0007", "val", "house_3", "000003", "push"),
    ]


def _splitter() -> OpenedWithGraspSplit:
    split = object.__new__(OpenedWithGraspSplit)
    split.dataset = Path("/tmp")
    split.val_fraction = 0.5
    split.seed = 0
    split.label = "opened_with_grasp"
    split.clips = _clips()
    return split


def test_house_mode_keeps_houses_disjoint():
    assigned = _splitter().assign(HOUSE, _clips())
    train = {clip.house for clip in assigned if clip.assigned_split == "train"}
    val = {clip.house for clip in assigned if clip.assigned_split == "val"}
    assert train.isdisjoint(val)
    assert train | val == {"house_1", "house_2", "house_3", "house_4"}
    assert all(clip.assigned_split in {"train", "val"} for clip in assigned)


def test_trajectory_mode_can_share_houses():
    clips = [
        SplitClip("0000", "train", "house_1", "000001", "push"),
        SplitClip("0001", "train", "house_1", "000002", "push"),
    ]
    assigned = _splitter().assign(TRAJECTORY, clips)
    train = {clip.house for clip in assigned if clip.assigned_split == "train"}
    val = {clip.house for clip in assigned if clip.assigned_split == "val"}
    assert train == {"house_1"}
    assert val == {"house_1"}
    assert sum(clip.assigned_split == "val" for clip in assigned) == 1
    assert sum(clip.assigned_split == "train" for clip in assigned) == 1


def test_nested_house_source_creates_linked_trajectory_split_and_is_protected(tmp_path):
    splitter = _splitter()
    splitter.dataset = tmp_path
    (tmp_path / "house/train").mkdir(parents=True)
    for clip in splitter.clips:
        folder = tmp_path / "house" / clip.source_split / clip.house
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"trajectories_batch_1_of_1_checkpoint_{clip.checkpoint}.h5").write_bytes(b"source")
    summary = splitter.write(TRAJECTORY)
    assert summary["counts"]["total"] == 8
    files = list((tmp_path / "trajectory").glob("*/house_*/*.h5"))
    assert len(files) == 8
    assert all(path.is_symlink() and path.read_bytes() == b"source" for path in files)
    with pytest.raises(ValueError, match="refusing"):
        splitter.write(HOUSE, tmp_path / "house")
    assert len(list((tmp_path / "house").glob("*/house_*/*.h5"))) == 8
