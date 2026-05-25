# RBY1 MolmoBot Evaluation Analysis

This note summarizes the current RBY1 MolmoBot-SPOC evaluation state and answers three practical questions:

1. What are the benchmark objects and why do labels like "puzzle" or "eye" look confusing?
2. How is the MolmoSpaces/MolmoBot dataset structured, and what would our lab data need to match?
3. Are the failed rollouts likely caused by code/config issues or by trained model behavior?

## 1. Object Labels And Intuitive Examples

The completed rollouts used the intended benchmark indices. The labels come from benchmark language fields and Objaverse metadata, not from our own naming.

### Completed Rollouts

`idx 0`:

```text
house: 1001
instruction: pick up the blue puzzle.
target: objapuzzlepiece_8882000196234ce7aefd64b149f3064e_1_0_4
referral: blue puzzle piece with rainbow
```

Saved log confirmation:

```text
Sampled task 'pick up the blue puzzle.'
completed ... object objapuzzlepiece_8882000196234ce7aefd64b149f3064e_1_0_4 ... success=False
```

Objaverse metadata for UID `8882000196234ce7aefd64b149f3064e`:

```text
category: puzzle piece
synset: jigsaw_puzzle.n.01
description: A pastel blue puzzle piece with a colorful rainbow and fluffy clouds.
```

`idx 2`:

```text
house: 1016
instruction: pick up the eye.
target: objaeyemodel_bba987e54f5d4724b9ffb799c8d98895_1_0_2
referral: eye model
```

Saved log confirmation:

```text
Sampled task 'pick up the eye.'
completed ... object objaeyemodel_bba987e54f5d4724b9ffb799c8d98895_1_0_2 ... success=False
```

Objaverse metadata for UID `bba987e54f5d4724b9ffb799c8d98895`:

```text
category: eye model
synset: model.n.04
description: A realistic human eye model with a detailed brown iris, prominent eyelashes, and smooth rounded back.
```

Conclusion: the benchmark index and target names are consistent. If the video looks like a dish or another object, the likely causes are noisy/ambiguous Objaverse visuals, camera viewpoint, scale, occlusion, or the policy moving near the wrong object. The benchmark language itself is metadata-derived and may not always match human visual intuition from the follower camera.

### Helper Script

Added:

```text
molmospaces/scripts/benchmarks/inspect_benchmark_episodes.py
```

Examples:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces
export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"

python scripts/benchmarks/inspect_benchmark_episodes.py \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --idx 0 2
```

Find more intuitive examples:

```bash
python scripts/benchmarks/inspect_benchmark_episodes.py \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --query cup bottle bowl banana plate toy \
  --unique_houses \
  --limit 12
```

Good candidate indices from the current benchmark:

```text
idx 9:  pick up the cup.                     house 1047
idx 29: pick up the green bottle.            house 1126
idx 31: pick up the toy robot.               house 1136
idx 33: pick up the turquoise spray bottle.  house 1138
idx 43: pick up the boat.                    house 1159
idx 63: pick up the clay cup.                house 1243
idx 69: pick up the bowl.                    house 1265
idx 79: pick up the cup.                     house 1336
```

Before running these, prepare assets for the chosen indices:

```bash
python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --idx 9 29 31
```

## 2. Dataset Structure, Benchmark Data, And Model Architecture

There are two related artifacts:

- `benchmark.json`: a simulation benchmark specification. It says how to reconstruct a fixed simulated task.
- H5 + MP4 rollout data: saved trajectories and videos produced by data generation or eval. This is also the style consumed by MolmoBot training loaders.

### A. Data Required To Build A MolmoSpaces Benchmark Episode

The benchmark JSON lives at:

```text
rby1_pick_benchmark/benchmark.json
```

Each entry is an `EpisodeSpec` defined in:

```text
molmospaces/molmo_spaces/evaluation/benchmark_schema.py
```

The key point is that benchmark construction needs simulation-reconstruction data: scene identity, assets, object poses, robot state, camera setup, task definition, and language.

```text
+====================================================================================================+
|                    DATA NEEDED TO BUILD A MOLMOSPACES BENCHMARK EPISODE                            |
+====================================================================================================+

1. SCENE / WORLD
----------------
+----------------------------------+
| scene_dataset                    |
| data_split                       |
| house_index                      |
| scene XML path or scene ID       |
+----------------------------------+

Meaning:
- Which simulated environment/house should be loaded?
- Example: procthor-objaverse / val / house 1016


2. OBJECT ASSETS
----------------
+----------------------------------+
| object mesh / XML asset path     |
| object asset ID                  |
| object body name in scene        |
| optional object texture/material |
+----------------------------------+

Meaning:
- What 3D objects exist in the scene?
- For Objaverse objects, this includes mesh files like visual.obj and collider.obj.
- The benchmark stores object names/poses; the MolmoSpaces resource manager resolves meshes/assets.


3. OBJECT POSES
---------------
+----------------------------------+
| object position                  |
| object orientation               |
| start pose of target object      |
| goal pose of target object       |
| poses of relevant scene objects  |
+----------------------------------+

Meaning:
- Where is each object at episode start?
- For pick tasks, the goal pose is usually the target object lifted upward.
- Stored in scene_modifications.object_poses, task.pickup_obj_start_pose, and task.pickup_obj_goal_pose.


4. ROBOT INITIAL STATE
----------------------
+----------------------------------+
| robot type / robot name          |
| initial joint positions          |
| robot base pose in world         |
| gripper state                    |
| head/torso/arm init qpos         |
+----------------------------------+

Meaning:
- Where is RBY1 placed?
- What joint configuration does it start with?
- Stored in robot.robot_name, robot.init_qpos, and task.robot_base_pose.


5. CAMERA SETUP
---------------
+----------------------------------+
| camera names                     |
| camera type                      |
| camera pose / mounting body      |
| camera intrinsics or fov         |
| image resolution                 |
| record_depth flag                |
+----------------------------------+

Meaning:
- What camera observations will the policy receive?
- For RBY1 rigid eval, key cameras are head_camera, wrist_camera_l, and wrist_camera_r.


6. TASK SPECIFICATION
---------------------
+----------------------------------+
| task class                       |
| task type                        |
| target object name               |
| success threshold                |
| task horizon                     |
+----------------------------------+

Meaning:
- What does the robot need to do?
- Example: task_cls = PickTask, task_type = pick, pickup_obj_name = target object.


7. LANGUAGE / INSTRUCTION
-------------------------
+----------------------------------+
| task_description                 |
| referral expressions             |
| alternative object labels        |
+----------------------------------+

Meaning:
- What instruction is given to the model?
- Example: "pick up the cup."
```

Compact view:

```text
+------------------+      +------------------+      +------------------+
| Scene Data       |      | Object Data      |      | Robot Data       |
+------------------+      +------------------+      +------------------+
| house index      |      | mesh / XML asset |      | robot name       |
| scene dataset    |      | object body name |      | init qpos        |
| split            |      | object pose      |      | base pose        |
| scene XML        |      | target object    |      | gripper state    |
+--------+---------+      +--------+---------+      +--------+---------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
                                   v
+------------------+      +------------------+      +------------------+
| Camera Data      |      | Task Data        |      | Language Data    |
+------------------+      +------------------+      +------------------+
| camera names     |      | task_cls         |      | instruction      |
| camera poses     |      | task_type        |      | object labels    |
| fov/intrinsics   |      | success rule     |      | referral exprs   |
| resolution       |      | horizon          |      |                  |
+--------+---------+      +--------+---------+      +--------+---------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
                                   v
                         +-------------------+
                         | benchmark.json    |
                         +-------------------+
                         | EpisodeSpec[0]    |
                         | EpisodeSpec[1]    |
                         | ...               |
                         +-------------------+
                                   |
                                   v
                         +-------------------+
                         | JsonEvalRunner    |
                         +-------------------+
                                   |
                                   v
                         +-------------------+
                         | Recreated Sim     |
                         | Evaluation Task   |
                         +-------------------+
```

For the current RBY1 pick benchmark, one episode has this shape:

```text
EpisodeSpec
├── source
├── house_index
├── scene_dataset
├── data_split
├── robot
│   ├── robot_name
│   └── init_qpos
├── img_resolution
├── cameras
├── scene_modifications
│   ├── added_objects
│   ├── object_poses
│   └── removed_objects
├── task
│   ├── task_cls
│   ├── robot_base_pose
│   ├── pickup_obj_name
│   ├── pickup_obj_start_pose
│   ├── pickup_obj_goal_pose
│   ├── succ_pos_threshold
│   ├── task_horizon_sec
│   └── task_type
├── task_relevant_objects
└── language
    ├── task_description
    ├── referral_expressions
    └── referral_expressions_priority
```

### B. How Benchmark JSON Is Used By MolmoBot-SPOC Eval

```text
+====================================================================================+
|                    MOLMOBOT-SPOC SIMULATION EVALUATION FLOW                        |
+====================================================================================+

  BENCHMARK DATA                     MOLMOSPACES SIM                      MOLMOBOT
  --------------                     ---------------                      --------

+----------------------+       +------------------------+        +----------------------+
| benchmark.json       |       | JsonEvalRunner          |        | RBY1RigidManipConfig |
|----------------------|       |------------------------|        |----------------------|
| house_index          | ----> | selects episode idx     |        | camera_names         |
| scene_dataset        |       | loads EpisodeSpec       |        | action_spec          |
| robot.init_qpos      |       | builds task sampler     |        | model checkpoint     |
| task.pickup_obj_name |       +-----------+------------+        +----------+-----------+
| language.task_desc   |                   |                                |
+----------------------+                   v                                v
                              +------------------------+        +----------------------+
                              | JsonEvalTaskSampler    |        | SPOCModelPolicy      |
                              |------------------------|        |----------------------|
                              | loads scene XML        |        | builds observation   |
                              | places objects         |        | preprocesses images  |
                              | places robot           |        | preprocesses text    |
                              | sets cameras           |        | builds proprio       |
                              | creates PickTask       |        +----------+-----------+
                              +-----------+------------+                   |
                                          |                                v
                                          |                    +----------------------+
                                          |                    | Spoc Model           |
                                          |                    |----------------------|
                                          |                    | image encoder        |
                                          |                    | text encoder         |
                                          |                    | action decoder       |
                                          |                    +----------+-----------+
                                          |                               |
                                          v                               v
+----------------------+       +------------------------+        +----------------------+
| H5 + MP4 output      | <---- | MolmoSpaces rollout    | <----  | action chunk         |
|----------------------|       |------------------------|        |----------------------|
| camera videos        |       | sim step               |        | base action          |
| actions              |       | robot control          |        | arm action           |
| qpos/qvel            |       | reward/success check   |        | gripper action       |
| success[]            |       | save trajectory        |        +----------------------+
| obs_scene            |       +------------------------+
+----------------------+
```

### C. Rollout / Training Data Format

Completed eval rollouts are saved under:

```text
MolmoBot/MolmoBot-SPOC/eval_output/RBY1RigidManipEvalConfig/<timestamp>/house_<id>/
```

Each completed rollout contains:

```text
trajectories_batch_1_of_1.h5
episode_00000000_camera_follower_batch_1_of_1.mp4
episode_00000000_head_camera_batch_1_of_1.mp4
episode_00000000_wrist_camera_l_batch_1_of_1.mp4
episode_00000000_wrist_camera_r_batch_1_of_1.mp4
```

The H5 schema is documented in:

```text
molmospaces/docs/data_format.md
```

Key groups observed in the saved RBY1 eval H5 files:

```text
traj_0/actions/commanded_action
traj_0/actions/joint_pos
traj_0/actions/joint_pos_rel
traj_0/actions/ee_pose
traj_0/actions/ee_twist
traj_0/obs/agent/qpos
traj_0/obs/agent/qvel
traj_0/obs/extra/task_info
traj_0/obs/extra/robot_base_pose
traj_0/obs/extra/tcp_pose
traj_0/obs/extra/obj_start
traj_0/obs/extra/obj_end
traj_0/obs/sensor_data/<camera_name>
traj_0/obs/sensor_param/<camera_name>/intrinsic_cv
traj_0/obs/sensor_param/<camera_name>/extrinsic_cv
traj_0/obs_scene
traj_0/rewards
traj_0/success
traj_0/terminated
traj_0/truncated
```

### D. H5 + MP4 Dataset To MolmoBot Model Architecture

MolmoBot training expects the H5/video style above. The training loader reference is:

```text
MolmoBot/MolmoBot/olmo/data/synthmanip_dataset.py
```

The RBY1 rigid model architecture/configuration is defined through:

```text
MolmoBot/MolmoBot-SPOC/eval/config/spoc_policy_configs.py
MolmoBot/MolmoBot-SPOC/architecture/__init__.py
MolmoBot/MolmoBot-SPOC/architecture/common/action_decoder.py
```

```text
+====================================================================================================+
|                          H5 + MP4 DATASET -> MOLMOBOT TRAINING EXAMPLE                             |
+====================================================================================================+

                                  one trajectory: traj_0
+----------------------------------------------------------------------------------------------------+
| trajectories_batch_1_of_1.h5                                                                       |
|                                                                                                    |
|  +----------------------------+       +----------------------------+       +---------------------+  |
|  | obs/sensor_data            |       | obs/agent                  |       | actions             |  |
|  |----------------------------|       |----------------------------|       |---------------------|  |
|  | head_camera  -> .mp4       |       | qpos                       |       | joint_pos_rel       |  |
|  | wrist_camera_l -> .mp4     |       | qvel                       |       | joint_pos           |  |
|  | wrist_camera_r -> .mp4     |       |                            |       | commanded_action    |  |
|  +-------------+--------------+       +-------------+--------------+       +----------+----------+  |
|                |                                    |                                 |             |
|                v                                    v                                 v             |
|  +----------------------------+       +----------------------------+       +---------------------+  |
|  | video frames               |       | proprio/state vector       |       | target action chunk |  |
|  | T x cameras x RGB          |       | qpos by move group         |       | future robot cmds   |  |
|  +-------------+--------------+       +-------------+--------------+       +----------+----------+  |
|                |                                    |                                 |             |
|                +------------------------+-----------+---------------------------------+             |
|                                         |                                                         |
|                                         v                                                         |
|  +---------------------------------------------------------------------------------------------+   |
|  | obs_scene                                                                                   |   |
|  |---------------------------------------------------------------------------------------------|   |
|  | task_description: "pick up the cup."                                                        |   |
|  | referral_expressions: object names / alternate labels                                       |   |
|  +----------------------------------------+----------------------------------------------------+   |
+-------------------------------------------|--------------------------------------------------------+
                                            |
                                            v
+====================================================================================================+
|                                      SynthmanipDataset                                             |
+====================================================================================================+
|                                                                                                    |
|  loads MP4 frames with decord                                                                      |
|  decodes qpos/qvel JSON                                                                            |
|  decodes action JSON                                                                               |
|  reads task_description                                                                            |
|  samples timestep and action horizon                                                               |
|                                                                                                    |
+-------------------+------------------------+-----------------------+-------------------------------+
                    |                        |                       |
                    v                        v                       v
        +----------------------+   +----------------------+   +----------------------+
        | image tensor         |   | proprio tensor       |   | language goal        |
        |----------------------|   |----------------------|   |----------------------|
        | head camera frames   |   | base                 |   | "pick up the cup."   |
        | wrist right frames   |   | left arm             |   +----------+-----------+
        | wrist left frames    |   | right arm            |              |
        +----------+-----------+   | grippers             |              |
                   |               +----------+-----------+              |
                   |                          |                          |
                   v                          v                          v
        +----------------------+   +----------------------+   +----------------------+
        | image preprocessing  |   | proprio projection   |   | text encoder         |
        |----------------------|   |----------------------|   |----------------------|
        | resize / normalize   |   | vector embedding     |   | goal_text_features   |
        | visual encoder       |   +----------+-----------+   +----------+-----------+
        | visual tokens        |              |                          |
        +----------+-----------+              |                          |
                   |                          |                          |
                   +--------------------------+--------------------------+
                                              |
                                              v
                              +-------------------------------+
                              | SpocContinuousActionModel     |
                              |-------------------------------|
                              | visual tokens                 |
                              | goal_text_features            |
                              | proprioception                |
                              | padding_mask                  |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | ParallelActionDecoder         |
                              |-------------------------------|
                              | learned action queries        |
                              | cross-attend to visual/text   |
                              | output binned action tokens   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | QuantileBinnedActionSpace     |
                              |-------------------------------|
                              | bins -> continuous actions    |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | predicted action chunk        |
                              |-------------------------------|
                              | base: 3                       |
                              | left_arm: 7                   |
                              | right_arm: 7                  |
                              | left_gripper: 1               |
                              | right_gripper: 1              |
                              | total: 19 dims                |
                              +-------------------------------+
```

Important indexing convention:

```text
state i corresponds to action i+1
frame/action 0 is dummy
last action is a done sentinel
```

## 3. Failure Diagnosis

The current failures are completed rollouts, not setup failures:

```text
Completed 1 houses, skipped 0 houses
Success count: 0, Total count: 1
```

That means the scene loaded, cameras initialized, model loaded, and the policy produced actions. This is different from earlier missing-asset failures, where the house was skipped and `Total count` was `0`.

### Correct Config/Data Combination

The eval uses the expected RBY1 rigid setup:

```text
config: RBY1RigidManipEvalConfig
model: allenai/MolmoBot-SPOC-RBY1Rigid
benchmark: molmospaces-bench-v2/.../rby1_benchmarks/pick_benchmark
camera: RBY1GoProD455CameraSystem
policy: SPOCModelPolicy
```

The model checkpoint is loaded from:

```text
~/.cache/huggingface/hub/models--allenai--MolmoBot-SPOC-RBY1Rigid/snapshots/01d1c5334e241c739099c2a043c5b93f87ee7eff/model.safetensors
```

### Pick Success Condition

`PickTask` success is defined in:

```text
molmospaces/molmo_spaces/tasks/pick_task.py
```

The target object must:

```text
1. be in contact only with robot geometry, and
2. be lifted at least succ_pos_threshold above its start height
```

For the tested episodes, `succ_pos_threshold` is `0.01` meters.

### Evidence From H5 Rollouts

`idx 0`:

```text
instruction: pick up the blue puzzle.
success any/last: False / False
max reward: 0.00283
last position_error: 0.2300
non-empty commanded_action steps: 200 / 201
left/right grasp states: only 0
```

Interpretation: the policy produced actions, but the object was never successfully lifted. The maximum lift-like reward was below the 1 cm threshold, and the final object position was farther from the goal pose.

`idx 2`:

```text
instruction: pick up the eye.
success any/last: False / False
max reward: 0.000001
last position_error: 0.0500
non-empty commanded_action steps: 200 / 201
left/right grasp states: only 0
```

Interpretation: the target object essentially stayed at the original 5 cm offset below the goal lift pose. The policy produced actions, but there is no evidence it grasped or lifted the target.

### Current Classification

Most likely: trained policy behavior / task difficulty / perception mismatch.

Less likely: code setup failure, because:

- benchmark-v2 scene loaded,
- target object existed,
- camera setup completed,
- checkpoint loaded,
- actions were generated for 200 steps,
- output H5 and videos were saved,
- no house was skipped.

Still possible and worth checking:

- visual target is ambiguous or misleading from camera views,
- language metadata is noisy,
- policy horizon of 20 seconds is too short for some scenes,
- RBY1 action scaling or command interpretation differs subtly from training assumptions.

## Recommended Next Runs

Use more visually intuitive examples first:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/molmospaces
conda activate mlspaces
export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"

python scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --idx 9 29 31
```

Then run MolmoBot eval, one index at a time:

```bash
cd /home/jinyoon/workspace/live-robotics-lab-project/MolmoBot/MolmoBot-SPOC
conda activate mlspaces

export PYTHONPATH="/home/jinyoon/workspace/live-robotics-lab-project/molmospaces:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu

python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 9
```

If the policy gets close but times out, rerun the same index with a longer horizon:

```bash
python -m molmo_spaces.evaluation.eval_main \
  molmobot_spoc.eval.config.rby1_eval_config:RBY1RigidManipEvalConfig \
  --benchmark_dir /home/jinyoon/workspace/live-robotics-lab-project/rby1_pick_benchmark \
  --no_wandb \
  --num_workers 1 \
  --idx 9 \
  --task_horizon_sec 40
```

## Practical Next Steps

1. Choose 3 to 5 intuitive indices using `inspect_benchmark_episodes.py`.
2. Prepare assets for those indices only.
3. Run one eval at a time and inspect videos.
4. For each completed rollout, inspect H5 `task_info`, `success`, rewards, and gripper state.
5. If failures persist on intuitive objects, treat this as likely model-quality or benchmark difficulty rather than setup failure.
