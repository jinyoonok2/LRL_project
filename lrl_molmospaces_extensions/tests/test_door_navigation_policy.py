"""Short navigation paths must remain valid cubic-spline inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import splprep

from lrl_molmospaces_extensions.policy.nav_then_door_opening_policy import (
    NavDoorAStarSmoothPlannerPolicy as LegacyNavigationPolicy,
)
from lrl_molmospaces_extensions.policy.navigation.door_navigation_policy import (
    NavDoorAStarSmoothPlannerPolicy,
)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_short_paths_can_build_cubic_splines(count):
    context = SimpleNamespace(
        robot_view=SimpleNamespace(base=SimpleNamespace(pose=np.eye(4))),
        target_object=SimpleNamespace(position=np.array([1.0, 0.0, 0.0])),
    )
    waypoints = np.column_stack((np.linspace(0.2, 1.0, count), np.zeros(count)))
    result = NavDoorAStarSmoothPlannerPolicy._ensure_spline_waypoints(context, waypoints)
    assert result.shape == (4, 2)
    assert np.isfinite(result).all()
    splprep(result.T, s=1e-5)


def test_existing_policy_import_is_preserved():
    assert LegacyNavigationPolicy is NavDoorAStarSmoothPlannerPolicy
