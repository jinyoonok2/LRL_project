from types import SimpleNamespace

import numpy as np

from lrl_molmospaces_extensions.baseline.config import BaselineConfig
from lrl_molmospaces_extensions.baseline.rerender import BaselineRerender
from lrl_molmospaces_extensions.rerender.slow_nav import CAMERAS


def test_rendered_pose_updates_points_camera_metadata_and_head_marker():
    rerender = object.__new__(BaselineRerender)
    rerender.baseline_config = BaselineConfig()
    rerender.object_names = {"door_handle": "selected_handle"}
    rerender.point_sizes = {("door_handle", "head_camera"): 10}
    pose = np.eye(4)
    pose[0, 3] = 2.0
    camera = SimpleNamespace(get_pose=lambda: pose, fov=60)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[10, 20] = 1
    requests = []
    def segmentation(body, **kwargs):
        requests.append(body)
        return mask
    env = SimpleNamespace(
        camera_manager=SimpleNamespace(registry={name: camera for name in CAMERAS}),
        get_segmentation_mask_of_object=segmentation,
    )
    images = {name: np.zeros((48, 48, 3), dtype=np.uint8) for name in CAMERAS}
    patches = rerender._frame_metadata(env, images)
    assert requests == ["selected_handle"]
    points = patches["/traj_0/obs/extra/object_image_points/door_handle/head_camera/points"]
    np.testing.assert_allclose(points[0], [20 / 48, 10 / 48])
    assert images["head_camera"][10, 20].tolist() == [0, 0, 255]
    assert not images["wrist_camera_l"].any()
    extrinsic = patches["/traj_0/obs/sensor_param/head_camera/extrinsic_cv"]
    assert extrinsic[0, 3] == -2.0
