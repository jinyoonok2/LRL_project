from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from lrl_molmospaces_extensions.dataset.episode import DoorOpeningEpisode
from lrl_molmospaces_extensions.dataset.numbered import NumberedClip, NumberedIndex


@dataclass(frozen=True)
class ClipCheck:
    number: str
    label: str | None
    split: str
    house: str
    checkpoint: str
    action: str
    success: bool
    grasp_frames: int
    first_success_frame: int | None
    first_success_s: float | None
    video: str
    h5: str


class OpenedWithoutGraspCheck:
    """Split successful door-open clips by whether the handle was actually held.

    Success is the recorded hinge flag. Grasp is both fingers on the handle.
    A short bump (fewer than min_grasp_frames) counts as opened without grasp.
    """

    OPENED_WITHOUT_GRASP = "opened_without_grasp"
    OPENED_WITH_GRASP = "opened_with_grasp"
    LABELS = (OPENED_WITHOUT_GRASP, OPENED_WITH_GRASP)

    def __init__(self, dataset: Path, min_grasp_frames: int = 30):
        self.dataset = dataset
        self.min_grasp_frames = min_grasp_frames

    @property
    def index(self) -> NumberedIndex:
        return NumberedIndex(self.dataset / "numbered" / "index.tsv")

    def label_episode(self, episode: DoorOpeningEpisode) -> str | None:
        # Production datagen saves only successful opens.
        if not episode.success:
            return None
        if episode.grasp_frames >= self.min_grasp_frames:
            return self.OPENED_WITH_GRASP
        return self.OPENED_WITHOUT_GRASP

    def inspect(self, clip: NumberedClip) -> ClipCheck:
        h5_path = clip.h5_path(self.dataset)
        episode = DoorOpeningEpisode.from_h5(h5_path)
        return ClipCheck(
            number=clip.number,
            label=self.label_episode(episode),
            split=clip.split,
            house=clip.house,
            checkpoint=clip.checkpoint,
            action=clip.action,
            success=episode.success,
            grasp_frames=episode.grasp_frames,
            first_success_frame=episode.first_success_frame,
            first_success_s=episode.first_success_s,
            video=str(clip.head_video_path(self.dataset)),
            h5=str(h5_path),
        )

    def run(self, output: Path | None = None) -> list[ClipCheck]:
        output = output or (self.dataset / "quality_check")
        results = [row for clip in self.index if (row := self.inspect(clip)).label]
        self._write(output, results)
        return results

    def _write(self, output: Path, results: list[ClipCheck]) -> None:
        leftover = output / "did_not_open"
        if leftover.exists():
            for old in leftover.glob("*.mp4"):
                old.unlink()
            leftover.rmdir()

        for label in self.LABELS:
            folder = output / label
            if folder.exists():
                for old in folder.glob("*.mp4"):
                    old.unlink()
            folder.mkdir(parents=True, exist_ok=True)

        counts = {label: 0 for label in self.LABELS}
        for row in results:
            counts[row.label] += 1
            src = Path(row.video)
            dst = output / row.label / f"{row.number}.mp4"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(src, dst)

        (output / "index.tsv").write_text(
            "number\tlabel\tsplit\thouse\tcheckpoint\taction\tsuccess\tgrasp_frames\tfirst_success_s\n"
            + "".join(
                f"{r.number}\t{r.label}\t{r.split}\t{r.house}\t{r.checkpoint}\t{r.action}\t"
                f"{int(r.success)}\t{r.grasp_frames}\t{'' if r.first_success_s is None else r.first_success_s}\n"
                for r in results
            )
        )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "dataset": str(self.dataset),
                    "min_grasp_frames": self.min_grasp_frames,
                    "counts": counts,
                    "clips": [asdict(row) for row in results],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"output": str(output), **counts}, indent=2), flush=True)
