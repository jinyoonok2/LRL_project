"""Opening-matched rerender with synchronized camera and point observations."""

import json

import cv2
import h5py
import numpy as np

from molmo_spaces.env.sensors_cameras import CameraParameterSensor

from lrl_molmospaces_extensions.baseline.points import visible_point
from lrl_molmospaces_extensions.rerender.opening_matched_nav import OpeningMatchedNavRerender
from lrl_molmospaces_extensions.rerender.slow_nav import CAMERAS


class BaselineRerender(OpeningMatchedNavRerender):
    """Final H5 and videos have prompts measured at each rendered robot pose."""

    def __init__(self, dataset, output, config):
        super().__init__(dataset, output)
        self.baseline_config = config

    def run(self):
        # The parent skips existing H5 files, even when a previous video write failed.
        # Refuse those partial artifacts instead of declaring them complete.
        for recording in self._iter_h5s():
            final = self._output_h5(recording)
            if not final.exists():
                continue
            with h5py.File(final, "r") as src:
                scene = json.loads(src["traj_0/obs_scene"][()].decode())
                videos = self._camera_filenames(src)
                complete = scene.get("baseline") == self.baseline_config.name
                complete = complete and all(
                    (path := self._output_video_path(final.parent, filename)).is_file()
                    and path.stat().st_size > 0
                    for filename in videos.values()
                )
            if not complete:
                raise RuntimeError(f"Partial rerender output: {final}; preserve it before retrying")
        return super().run()

    def _rewrite_trajectory(self, env, defaults, h5_path):
        with h5py.File(h5_path, "r") as src:
            scene = json.loads(src["traj_0/obs_scene"][()].decode())
            self.object_names = scene.get("task_object_names", {})
            self.point_sizes = {}
            group = src.get("traj_0/obs/extra/object_image_points")
            if self.baseline_config.point_prompt:
                if not self.object_names.get("door_handle") or group is None:
                    raise ValueError(f"Missing task_object_names/point observations in {h5_path}")
                for object_key in group:
                    if object_key not in self.object_names:
                        raise ValueError(f"Missing body name for {object_key} in {h5_path}")
                    for camera in group[object_key]:
                        self.point_sizes[object_key, camera] = group[object_key][camera]["points"].shape[1]
        result = super()._rewrite_trajectory(env, defaults, h5_path)
        with h5py.File(result["output"], "r+") as dst:
            scene["baseline"] = self.baseline_config.name
            scene["point_prompt_mode"] = "per_frame" if self.baseline_config.point_prompt else "none"
            scene["point_marker_in_video"] = self.baseline_config.draw_point
            scene["navigation_speed"] = "opening_matched"
            del dst["traj_0/obs_scene"]
            dst["traj_0/obs_scene"] = json.dumps(scene).encode()
        return result

    def _frame_metadata(self, env, images):
        patches = {}
        for camera in CAMERAS:
            height, width = images[camera].shape[:2]
            params = CameraParameterSensor(camera, img_resolution=(height, width)).get_observation(env, None)
            for key, value in params.items():
                patches[f"/traj_0/obs/sensor_param/{camera}/{key}"] = np.asarray(value)
        if not self.baseline_config.point_prompt:
            return patches
        for (object_key, camera), max_points in self.point_sizes.items():
            mask = env.get_segmentation_mask_of_object(
                self.object_names[object_key], camera_name=camera, batch_index=0
            )
            if mask is None:
                mask = np.zeros(images[camera].shape[:2], dtype=np.uint8)
            points, count = visible_point(mask, max_points)
            prefix = f"/traj_0/obs/extra/object_image_points/{object_key}/{camera}"
            patches[f"{prefix}/points"] = points
            patches[f"{prefix}/num_points"] = count
            if object_key == "door_handle" and camera == "head_camera" and count[0] and self.baseline_config.draw_point:
                height, width = images[camera].shape[:2]
                pixel = tuple(np.rint(points[0] * [width, height]).astype(int))
                cv2.circle(images[camera], pixel, self.baseline_config.marker_radius + 2, (255, 255, 255), -1)
                cv2.circle(images[camera], pixel, self.baseline_config.marker_radius, (0, 0, 255), -1)
        return patches
