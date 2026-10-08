"""Train/val splits over opened-with-grasp trajectories.

House mode keeps every house in one split. Trajectory mode splits individual
clips, so the same house can appear in both train and val.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

HOUSE = "house"
TRAJECTORY = "trajectory"
MODES = (HOUSE, TRAJECTORY)
DEFAULT_DATASETS = (
    Path("/bigtemp/xna8aw/nav_door/data/final"),
)


@dataclass(frozen=True)
class SplitClip:
    number: str
    source_split: str
    house: str
    checkpoint: str
    action: str
    assigned_split: str = ""


class OpenedWithGraspSplit:
    """Build a MolmoBot train/val tree from opened-with-grasp clips."""

    def __init__(
        self,
        dataset: Path,
        val_fraction: float = 0.15,
        seed: int = 42,
        label: str = "opened_with_grasp",
    ):
        self.dataset = dataset
        self.val_fraction = val_fraction
        self.seed = seed
        self.label = label
        self.clips = self._load_clips()

    def _load_clips(self) -> list[SplitClip]:
        path = self.dataset / "quality_check" / "index.tsv"
        clips = []
        for line in path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            number, label, split, house, checkpoint, action, *_ = line.split("\t")
            if label != self.label:
                continue
            clips.append(
                SplitClip(
                    number=number,
                    source_split=split,
                    house=house,
                    checkpoint=checkpoint,
                    action=action,
                )
            )
        if not clips:
            raise ValueError(f"no {self.label} clips in {path}")
        return clips

    @property
    def source_root(self) -> Path:
        """The retained house split owns files; other splits link to it."""
        if (self.dataset / "house" / "train").is_dir():
            return self.dataset / "house"
        return self.dataset

    def assign(self, mode: str, clips: list[SplitClip] | None = None) -> list[SplitClip]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        clips = list(clips if clips is not None else self.clips)
        if mode == HOUSE:
            return self._assign_houses(clips)
        return self._assign_trajectories(clips)

    def _assign_houses(self, clips: list[SplitClip]) -> list[SplitClip]:
        by_house: dict[str, list[SplitClip]] = defaultdict(list)
        for clip in clips:
            by_house[clip.house].append(clip)
        houses = []
        for house in sorted(by_house):
            actions = Counter(clip.action for clip in by_house[house])
            majority = "push" if actions["push"] >= actions["pull"] else "pull"
            houses.append((house, majority))
        train_houses, val_houses = self._stratified(houses, key=lambda item: item[1])
        assigned = {house: "train" for house, _ in train_houses}
        assigned.update({house: "val" for house, _ in val_houses})
        return [self._with_split(clip, assigned[clip.house]) for clip in clips]

    def _with_split(self, clip: SplitClip, assigned_split: str) -> SplitClip:
        return SplitClip(
            number=clip.number,
            source_split=clip.source_split,
            house=clip.house,
            checkpoint=clip.checkpoint,
            action=clip.action,
            assigned_split=assigned_split,
        )

    def _assign_trajectories(self, clips: list[SplitClip]) -> list[SplitClip]:
        train, val = self._stratified(clips, key=lambda clip: clip.action)
        return [self._with_split(clip, "train") for clip in train] + [
            self._with_split(clip, "val") for clip in val
        ]

    def _stratified(self, items: list, key) -> tuple[list, list]:
        rng = random.Random(self.seed)
        buckets: dict[str, list] = defaultdict(list)
        for item in items:
            buckets[str(key(item))].append(item)
        train: list = []
        val: list = []
        for group_key in sorted(buckets):
            group = list(buckets[group_key])
            rng.shuffle(group)
            n_val = int(round(len(group) * self.val_fraction))
            n_val = max(0, min(n_val, len(group)))
            if n_val == len(group) and len(group) > 1:
                n_val -= 1
            val.extend(group[:n_val])
            train.extend(group[n_val:])
        return train, val

    def write(self, mode: str, output: Path | None = None) -> dict:
        assigned = self.assign(mode)
        default = self.dataset / mode if self.source_root != self.dataset else self.dataset / "opened_with_grasp" / mode
        output = (output or default).resolve()
        self._assert_safe_output(output)
        self._reset(output)
        for clip in assigned:
            dest_house = output / clip.assigned_split / clip.house
            for src, rel in self._files_for(clip):
                dest = dest_house / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(src.resolve())
        summary = self._summary(mode, output, assigned)
        (output / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "index.tsv").write_text(
            "number\tsplit\thouse\tcheckpoint\taction\tsource_split\n"
            + "".join(
                f"{c.number}\t{c.assigned_split}\t{c.house}\t{c.checkpoint}\t"
                f"{c.action}\t{c.source_split}\n"
                for c in sorted(assigned, key=lambda clip: clip.number)
            )
        )
        print(json.dumps({k: summary[k] for k in ("output", "mode", "counts")}, indent=2), flush=True)
        return summary

    def run(self, modes: tuple[str, ...] = MODES) -> list[dict]:
        return [self.write(mode) for mode in modes]

    def _files_for(self, clip: SplitClip) -> list[tuple[Path, Path]]:
        house_dir = self.source_root / clip.source_split / clip.house
        pairs = []
        pattern = f"*checkpoint_{clip.checkpoint}.*"
        for src in sorted(house_dir.glob(pattern)):
            if src.is_file():
                pairs.append((src, Path(src.name)))
        other = house_dir / "other_cameras"
        if other.is_dir():
            for src in sorted(other.glob(pattern)):
                if src.is_file():
                    pairs.append((src, Path("other_cameras") / src.name))
        if not pairs:
            raise FileNotFoundError(
                f"no files for {clip.number} {clip.house} checkpoint {clip.checkpoint} in {house_dir}"
            )
        return pairs

    def _assert_safe_output(self, output: Path) -> None:
        forbidden = {
            self.dataset.resolve(),
            (self.dataset / "train").resolve(),
            (self.dataset / "val").resolve(),
            self.source_root.resolve(),
            (self.source_root / "train").resolve(),
            (self.source_root / "val").resolve(),
        }
        if output in forbidden:
            raise ValueError(f"refusing to overwrite original dataset paths: {output}")

    @staticmethod
    def _reset(output: Path) -> None:
        for split in ("train", "val"):
            folder = output / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)

    def _summary(self, mode: str, output: Path, assigned: list[SplitClip]) -> dict:
        counts = Counter(clip.assigned_split for clip in assigned)
        houses = {
            split: sorted({clip.house for clip in assigned if clip.assigned_split == split})
            for split in ("train", "val")
        }
        overlap = sorted(set(houses["train"]) & set(houses["val"]))
        return {
            "output": str(output),
            "dataset": str(self.dataset),
            "mode": mode,
            "label": self.label,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "counts": {
                "train": counts["train"],
                "val": counts["val"],
                "total": len(assigned),
            },
            "houses": houses,
            "overlapping_houses": overlap,
            "clips": [asdict(clip) for clip in assigned],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make train/val trees from opened-with-grasp trajectories."
    )
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = args.dataset or list(DEFAULT_DATASETS)
    for dataset in datasets:
        splitter = OpenedWithGraspSplit(
            dataset, val_fraction=args.val_fraction, seed=args.seed
        )
        modes = tuple(args.mode) if args.mode else ((TRAJECTORY,) if splitter.source_root != dataset else MODES)
        splitter.run(modes)


if __name__ == "__main__":
    main()
