# RB-Y1 Manual Teleoperation: Local Setup and Run Guide

## Purpose

This branch is dedicated to manually controlling the simulated RB-Y1 in MolmoSpaces and recording human demonstrations. The first target is door opening with a locally connected SpaceMouse.

The intended data flow is:

```text
Local SpaceMouse
  → RB-Y1 teleoperation policy
  → MolmoSpaces / MuJoCo simulation
  → H5 trajectory and camera videos
```

MolmoBot is not required for manual control. A*, cuRobo, and the learned VLA are not used to generate the human actions.

## Current Status

MolmoSpaces already includes generic keyboard, SpaceMouse, and phone policies. The generic policies assume a single `arm` and `gripper`, so they are not yet suitable for RB-Y1 door manipulation.

The next branch implementation must add an RB-Y1 adapter with:

- Mobile-base movement.
- Active arm selection between left and right.
- Cartesian end-effector translation and rotation.
- Independent left/right gripper control.
- Optional torso and head control.
- MuJoCo viewer integration.
- H5 and camera-video recording through the existing data-generation pipeline.

Do not use the generic `run_fast_teleop.py --robot rby1` command for full door manipulation until this adapter is implemented.

## Recommended Local Hardware

- Ubuntu 22.04 or another recent Linux distribution.
- NVIDIA GPU with working drivers and OpenGL support.
- At least 8GB GPU memory for interactive simulation.
- At least 16GB system RAM; 32GB is preferred.
- Sufficient storage for MolmoSpaces assets and recorded videos.
- A locally connected SpaceMouse.

The SpaceMouse must be attached to the same computer running the teleoperation policy. SSH does not normally forward USB HID devices.

## Clone the Manual-Teleoperation Branch

```bash
git clone --recurse-submodules \
  --branch rby1-manual-teleop \
  https://github.com/jinyoonok2/LRL_project.git

cd LRL_project
git submodule update --init --recursive
```

If the repository was already cloned:

```bash
git fetch origin
git switch rby1-manual-teleop
git submodule update --init --recursive
```

## Create the Python Environment

Install Miniconda or Miniforge, then run:

```bash
conda create -n molmospaces-teleop python=3.11 -y
conda activate molmospaces-teleop

cd molmospaces
pip install --no-cache-dir -e ".[mujoco]"
```

Install SpaceMouse system and Python support:

```bash
sudo apt-get update
sudo apt-get install -y libhidapi-hidraw0
pip install hidapi pynput
```

cuRobo is not required for human-controlled trajectories. Install the `curobo` extra only if planner comparison is needed:

```bash
pip install --no-cache-dir -e ".[curobo]"
```

## Configure SpaceMouse Permissions

Create a udev rule:

```bash
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/99-spacemouse.rules

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the SpaceMouse, then verify detection:

```bash
python - <<'PY'
import hid

for device in hid.enumerate(0x256F, 0):
    print(
        hex(device["vendor_id"]),
        hex(device["product_id"]),
        device.get("product_string"),
    )
PY
```

Common product IDs in the existing policy:

- `50741`: wireless SpaceMouse.
- `50734`: wired SpaceMouse.

## Configure Interactive MuJoCo Rendering

The local session must have a graphical display:

```bash
echo "$DISPLAY"
```

For a local desktop, use GLFW instead of the headless EGL settings used on Slurm:

```bash
export MUJOCO_GL=glfw
unset PYOPENGL_PLATFORM
unset MUJOCO_EGL_DEVICE_ID
```

Verify MuJoCo:

```bash
python - <<'PY'
import mujoco
import mujoco.viewer

print("MuJoCo version:", mujoco.__version__)
print("Interactive viewer import succeeded")
PY
```

## Install MolmoSpaces Resources

From the `molmospaces` directory:

```bash
python -m molmo_spaces.molmo_spaces_constants
```

Keep large resource caches outside small home-directory quotas:

```bash
export MLSPACES_CACHE_DIR="/path/with/free/space/molmo-spaces-resources"
export MLSPACES_ASSETS_DIR="/path/with/free/space/molmospaces-assets"
export HF_HOME="/path/with/free/space/huggingface"
export TORCH_HOME="/path/with/free/space/torch"
export XDG_CACHE_HOME="/path/with/free/space/xdg"
```

## Planned RB-Y1 SpaceMouse Controls

- SpaceMouse translation: move the active end effector in XYZ.
- SpaceMouse rotation: rotate the active end effector in roll, pitch, and yaw.
- Right button: enable/disable motion.
- Left button: toggle the active gripper.
- Keyboard `1`: select the left arm.
- Keyboard `2`: select the right arm.
- Keyboard `W/A/S/D`: move the mobile base.
- Keyboard `Q/E`: rotate the mobile base.
- Keyboard `R`: reset the episode.
- Keyboard `Esc`: end and save the episode.

The final key map may change during implementation and testing.

## Planned Run Command

After the RB-Y1 adapter is implemented:

```bash
cd molmospaces

python scripts/datagen/run_rby1_door_teleop.py \
  --device spacemouse \
  --house-index 22 \
  --viewer \
  --record
```

Expected output:

```text
molmospaces/assets/experiment_output/datagen/rby1_manual_door_teleop/
  house_<id>/
    trajectories_batch_*.h5
    episode_*_head_camera_*.mp4
    episode_*_wrist_camera_*.mp4
```

## Validation Checklist

Before recording demonstrations:

- SpaceMouse is detected through `hid.enumerate`.
- MuJoCo viewer opens and updates smoothly.
- The selected arm follows Cartesian commands.
- Base and arm controls do not overwrite each other.
- Gripper open/close state is correct.
- Reset restores the initial door and robot state.
- Ending an episode saves H5 and video files.
- Saved trajectories pass `scripts/data/validate_trajectories.py`.

## Next Development Step

Implement `RBY1SpaceMouseTeleopPolicy` and a door-teleoperation launcher, then validate base-only movement before enabling arm and gripper control.
