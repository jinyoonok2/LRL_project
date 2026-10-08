from pathlib import Path

from lrl_molmospaces_extensions.dataset.episode import DoorOpeningEpisode
from lrl_molmospaces_extensions.quality_check.opened_without_grasp import (
    OpenedWithoutGraspCheck,
)


def _episode(success: bool, grasp_frames: int) -> DoorOpeningEpisode:
    return DoorOpeningEpisode(
        path=Path("unused.h5"),
        frames=10,
        success=success,
        first_success_frame=5 if success else None,
        grasp_frames=grasp_frames,
        first_grasp_frame=0 if grasp_frames else None,
        last_grasp_frame=3 if grasp_frames else None,
    )


def test_opened_without_grasp_labels():
    check = OpenedWithoutGraspCheck(Path("/tmp"), min_grasp_frames=30)
    assert check.label_episode(_episode(True, 0)) == check.OPENED_WITHOUT_GRASP
    assert check.label_episode(_episode(True, 14)) == check.OPENED_WITHOUT_GRASP
    assert check.label_episode(_episode(True, 90)) == check.OPENED_WITH_GRASP
    assert check.label_episode(_episode(False, 0)) is None
