import json

import numpy as np
import pytest

from lrl_molmospaces_extensions.baseline.config import BaselineConfig
from lrl_molmospaces_extensions.baseline.points import visible_point
from lrl_molmospaces_extensions.baseline.runner import BaselinePipeline


def test_prompt_is_on_visible_pixel_and_empty_mask_has_no_prompt():
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 4:7] = 1
    points, count = visible_point(mask, 10)
    assert count.tolist() == [1]
    np.testing.assert_allclose(points[0], [0.5, 3 / 8])
    assert np.isnan(points[1:]).all()
    points, count = visible_point(np.zeros_like(mask), 10)
    assert count.tolist() == [0]
    assert np.isnan(points).all()


@pytest.mark.parametrize("options", [
    {"name": "../escape"},
    {"house_inds": []},
    {"val_house_inds": [99]},
    {"successes_per_door": 3, "max_attempts_per_door": 2},
    {"point_prompt": False, "draw_point": True},
])
def test_invalid_baseline_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        BaselineConfig(**options)


def test_shards_preserve_the_declared_house_holdout():
    config = BaselineConfig(house_inds=list(range(100)), val_house_inds=[1, 22, 99])
    shard = config.for_shard("b")
    assert shard.house_inds == list(range(11, 23))
    assert shard.val_house_inds == [22]


def test_recordings_are_house_split_without_moving_raw_data(tmp_path):
    config = BaselineConfig(output_root=str(tmp_path), house_inds=[1, 2], val_house_inds=[2])
    pipeline = BaselinePipeline(config, "test")
    for house in [1, 2]:
        folder = pipeline.root / "raw" / f"house_{house}"
        folder.mkdir(parents=True)
        (folder / "episode.h5").write_bytes(b"recording")
    assert pipeline.organize()["trajectories"] == 2
    assert (pipeline.root / "recordings/train/house_1").is_symlink()
    assert (pipeline.root / "recordings/val/house_2").is_symlink()
    assert (pipeline.root / "raw/house_2/episode.h5").read_bytes() == b"recording"


def test_failed_pipeline_resumes_completed_stages_and_rejects_config_changes(tmp_path, monkeypatch):
    config = BaselineConfig(output_root=str(tmp_path), results_root=str(tmp_path / "results"))
    pipeline = BaselinePipeline(config, "test")
    calls = []
    for stage in ("generate", "organize", "validate"):
        monkeypatch.setattr(pipeline, stage, lambda stage=stage: calls.append(stage) or {})
    def fail():
        raise RuntimeError("render interrupted")
    monkeypatch.setattr(pipeline, "rerender", fail)
    with pytest.raises(RuntimeError):
        pipeline.run()
    state = json.loads(pipeline.manifest.read_text())
    assert state["completed_stages"] == ["generate", "organize"]
    assert state["status"] == "failed"
    monkeypatch.setattr(pipeline, "rerender", lambda: calls.append("rerender") or {})
    assert pipeline.run(resume=True)["status"] == "complete"
    assert calls == ["generate", "organize", "rerender", "validate"]
    changed = BaselinePipeline(BaselineConfig(output_root=str(tmp_path), results_root=str(tmp_path / "results"), seed=7), "test")
    with pytest.raises(ValueError, match="differs"):
        changed.run(resume=True)
    with pytest.raises(FileExistsError):
        pipeline.run()
