"""Door-target navigation adapter with safe short-path spline construction."""

import logging

import numpy as np
from scipy.interpolate import splev, splprep

from lrl_molmospaces_extensions.policy.navigation.astar_planner_policy import (
    AStarSmoothPlannerPolicy,
)

log = logging.getLogger(__name__)


class NavDoorAStarSmoothPlannerPolicy(AStarSmoothPlannerPolicy):
    """Door-opening navigation variant that guarantees valid spline inputs."""

    def _ensure_spline_waypoints(self, world_waypoints: np.ndarray) -> np.ndarray:
        if len(world_waypoints) >= 4:
            return world_waypoints

        if len(world_waypoints) == 0:
            start_xy = self.robot_view.base.pose[:2, 3]
            target_xy = self.target_object.position[:2]
            world_waypoints = np.stack([start_xy, target_xy])
        elif len(world_waypoints) == 1:
            start_xy = self.robot_view.base.pose[:2, 3]
            goal_xy = world_waypoints[0]
            if np.linalg.norm(goal_xy - start_xy) < 1e-3:
                target_xy = self.target_object.position[:2]
                direction = goal_xy - target_xy
                norm = np.linalg.norm(direction)
                if norm < 1e-3:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                start_xy = goal_xy - direction * 0.1
            world_waypoints = np.stack([start_xy, goal_xy])

        deltas = np.diff(world_waypoints, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]
        if total_length < 1e-6:
            offsets = np.linspace(-0.05, 0.05, 4)
            return world_waypoints[-1][None, :] + np.stack(
                [offsets, np.zeros_like(offsets)], axis=1
            )

        sample_lengths = np.linspace(0.0, total_length, 4)
        x = np.interp(sample_lengths, cumulative_lengths, world_waypoints[:, 0])
        y = np.interp(sample_lengths, cumulative_lengths, world_waypoints[:, 1])
        log.info(
            "[NAV-DOOR A* SMOOTH] Upsampled %d path waypoints to 4 spline waypoints.",
            len(world_waypoints),
        )
        return np.stack([x, y], axis=1)

    def build_policy_plan(self, world_waypoints):
        world_waypoints = self.stop_plan(world_waypoints)
        world_waypoints = self._ensure_spline_waypoints(world_waypoints)

        plan_length = sum(
            np.linalg.norm(world_waypoints[it] - world_waypoints[it - 1])
            for it in range(1, len(world_waypoints))
        )
        num_points = max(
            2,
            int(np.ceil(plan_length / self.config.policy_config.path_max_inter_waypoint_dist)),
        )

        tck, u = splprep(world_waypoints.transpose(), s=1e-5)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        dx_du, dy_du = splev(u_new, tck, der=1)
        thetas = np.arctan2(dy_du, dx_du)

        combined_waypoints = []
        start_theta = float(
            np.asarray(self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]).reshape(-1)[0]
        )
        for theta in self.max_angle_waypoints(
            np.stack([start_theta, float(thetas[0])])[:, None]
        ):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for cur_x, cur_y, cur_theta in zip(x_new, y_new, thetas):
            combined_waypoints.append(np.stack((cur_x, cur_y, cur_theta)))

        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], final_theta])[:, None]):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)
