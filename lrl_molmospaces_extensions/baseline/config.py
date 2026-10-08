"""One configuration for generation and final dataset preparation."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import re

import yaml

SHARDS = {"a": (0, 11), "b": (11, 23), "c": (23, 42), "d": (42, 61), "e": (61, 80), "f": (80, 100)}


@dataclass(frozen=True)
class BaselineConfig:
    name: str = "nav_door_point_matched"
    output_root: str = "/bigtemp/xna8aw/nav_door/baselines"
    results_root: str = "/u/xna8aw/workspace/live-robotics-lab/LRL_project/results"
    generation_config: str = (
        "lrl_molmospaces_extensions.configs.door_opening:"
        "RBY1NavDoorOpeningSimpleVisibleProductionBaseConfig"
    )
    house_inds: list[int] = field(default_factory=lambda: [22])
    val_house_inds: list[int] = field(default_factory=list)
    successes_per_door: int = 1
    max_attempts_per_door: int = 2
    required_action: str = "both"
    seed: int = 42
    point_prompt: bool = True
    draw_point: bool = True
    marker_radius: int = 5

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.name):
            raise ValueError("name must be a single folder name")
        if not self.house_inds or any(h < 0 for h in self.house_inds):
            raise ValueError("house_inds must contain nonnegative house IDs")
        if len(set(self.house_inds)) != len(self.house_inds):
            raise ValueError("house_inds must be unique")
        if not set(self.val_house_inds) <= set(self.house_inds):
            raise ValueError("validation houses must be included in house_inds")
        if self.required_action not in {"both", "push", "pull"}:
            raise ValueError("required_action must be both, push, or pull")
        if not 1 <= self.successes_per_door <= self.max_attempts_per_door:
            raise ValueError("attempt budget must cover the success target")
        if self.marker_radius < 1 or (self.draw_point and not self.point_prompt):
            raise ValueError("drawing requires point_prompt and a positive marker radius")

    @classmethod
    def load(cls, path: Path):
        return cls(**yaml.safe_load(path.read_text()))

    def to_dict(self):
        return asdict(self)

    def for_shard(self, shard: str):
        start, stop = SHARDS[shard]
        houses = [house for house in self.house_inds if start <= house < stop]
        return replace(self, house_inds=houses, val_house_inds=[house for house in self.val_house_inds if house in houses])

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
            raise ValueError("run_id must be a single folder name")
        return Path(self.output_root) / self.name / run_id
