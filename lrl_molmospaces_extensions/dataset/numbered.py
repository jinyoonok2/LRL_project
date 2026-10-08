from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NumberedClip:
    number: str
    split: str
    house: str
    checkpoint: str
    action: str
    description: str

    @property
    def house_id(self) -> int:
        return int(self.house.split("_")[1])

    def h5_path(self, dataset: Path) -> Path:
        root = dataset / "house" if (dataset / "house" / "train").is_dir() else dataset
        return root / self.split / self.house / f"trajectories_batch_1_of_1_checkpoint_{self.checkpoint}.h5"

    def head_video_path(self, dataset: Path) -> Path:
        numbered = dataset / "numbered" / f"{self.number}.mp4"
        if numbered.is_file():
            return numbered
        root = dataset / "house" if (dataset / "house" / "train").is_dir() else dataset
        return (
            root
            / self.split
            / self.house
            / f"episode_00000000_head_camera_batch_1_of_1_checkpoint_{self.checkpoint}.mp4"
        )


class NumberedIndex:
    def __init__(self, path: Path):
        self.path = path
        self.clips = self._load(path)

    @staticmethod
    def _load(path: Path) -> list[NumberedClip]:
        clips = []
        for line in path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            number, split, house, checkpoint, action, description = line.split("\t")
            clips.append(
                NumberedClip(
                    number=number,
                    split=split,
                    house=house,
                    checkpoint=checkpoint,
                    action=action,
                    description=description,
                )
            )
        return clips

    def __iter__(self):
        return iter(self.clips)

    def __len__(self) -> int:
        return len(self.clips)
