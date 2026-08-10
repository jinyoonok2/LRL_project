# RB-Y1 MolmoBot Evaluation Compatibility

For anyone starting from a fresh MolmoBot setup and trying to run the released RB-Y1 MolmoBot checkpoint with MolmoSpaces, there are two separate compatibility points to watch:

1. Do not assume latest MolmoSpaces `main` works with the released MolmoBot RB-Y1 checkpoint.
2. Do not assume the benchmark dataset version requested by the pinned MolmoSpaces commit is sufficient for RB-Y1 eval.

The setup that worked is:

- Start from the MolmoSpaces code version pinned by MolmoBot.
- Use the RB-Y1 benchmark resources from `molmospaces-bench-v2/20260327`.
- Apply the RB-Y1 Pick/PnP evaluation fixes from `allenai/molmospaces#135`.

Related links:

```text
MolmoBot issue:
https://github.com/allenai/MolmoBot/issues/9

MolmoSpaces PR for RB-Y1 Pick/PnP success-check fixes:
https://github.com/allenai/molmospaces/pull/135

MolmoSpaces commit associated with the RB-Y1 benchmark-version bump:
https://github.com/allenai/molmospaces/commit/e1a2fe16
```

## Compatibility Summary

| Area | Version / File | What To Do | Why It Matters |
|---|---|---|---|
| MolmoSpaces base code version | `allenai/molmospaces@cd23becebcf72dd93a4aa5872a60802d5eff03ef` | Start from the MolmoSpaces commit pinned by the released MolmoBot package. | Latest MolmoSpaces `main` may not be compatible with the released RB-Y1 MolmoBot checkpoint/eval code. |
| MolmoBot pin location | `MolmoBot/MolmoBot/pyproject.toml` | Check the MolmoSpaces dependency pin here. | This tells you which MolmoSpaces code version the released MolmoBot package expects. |
| RB-Y1 benchmark dataset version | `molmospaces-bench-v2/20260327` | Use this benchmark resource version for RB-Y1 evaluation. | The four RB-Y1 benchmarks are documented under this benchmark release and spread across three scene datasets. |
| Dataset bump reference | `allenai/molmospaces@e1a2fe16` | Use the RB-Y1 benchmark paths/version introduced by this MolmoSpaces update. | The pinned MolmoSpaces commit requests `20260325_1`, but that release does not contain the needed `rby1_benchmarks/` directories. |
| Latest MolmoSpaces `main` dataset request | `molmospaces-bench-v2/20260415` | Do not rely on this for public reproduction unless the dataset is available. | MolmoSpaces `main` requests `20260415`, but that benchmark version was not published on the public Hugging Face dataset when checked. |
| PR reference | `allenai/molmospaces#135` | Apply the RB-Y1 Pick/PnP evaluation compatibility fixes from this PR on top of the compatible MolmoSpaces code. | These fixes correct RB-Y1-specific contact-classification behavior in success checking. |
| First PR fix | `8797f8818086174509d3a9c288eccbaf713ad532` | In `rby1_view.py`, make RB-Y1 gripper EE and holo base `root_body_id` values use `.id`. | Prevents MuJoCo body view objects from being compared against integer `body_rootid` values. |
| Second PR fix | `6ebeeb8a38a6899b583e98c5c59cb25ccab76efe` | In `pick_task.py` and `pick_and_place_task.py`, compare contacts against the robot kinematic tree root via `data.model.body_rootid[int(base.root_body_id)]`. | RB-Y1's base body is nested under an outer kinematic-tree root, so direct comparison against `base.root_body_id` can misclassify robot-object contacts. |

## Required Benchmark Dataset Version

For RB-Y1 MolmoBot evaluation, use:

```text
molmospaces-bench-v2/20260327
```

The four RB-Y1 benchmark families are spread across three scene datasets:

| Task Family | Benchmark Dataset Version | Scene Dataset | Benchmark Directory |
|---|---|---|---|
| Pick | `molmospaces-bench-v2/20260327` | `procthor-objaverse` | `procthor-objaverse/rby1_benchmarks/pick_benchmark` |
| Pick-and-Place | `molmospaces-bench-v2/20260327` | `procthor-objaverse` | `procthor-objaverse/rby1_benchmarks/pnp_benchmark` |
| Opening | `molmospaces-bench-v2/20260327` | `ithor` | `ithor/rby1_benchmarks/opening_benchmark` |
| Door Opening | `molmospaces-bench-v2/20260327` | `procthor-10k` | `procthor-10k/rby1_benchmarks/door_opening_benchmark` |

Important dataset-version notes:

1. Latest MolmoSpaces `main` requests:

```text
molmospaces-bench-v2/20260415
```

but that benchmark version was not available on the public Hugging Face dataset when checked.

2. The MolmoSpaces commit pinned by the released MolmoBot package requests:

```text
molmospaces-bench-v2/20260325_1
```

but that benchmark version does not contain the needed RB-Y1 `rby1_benchmarks/` directories.

3. Therefore, for RB-Y1 MolmoBot evaluation, the benchmark resources should be bumped to:

```text
molmospaces-bench-v2/20260327
```

This dataset/path update is associated with MolmoSpaces commit:

```text
e1a2fe16
```

## Step-by-Step For A Fresh Setup

### 1. Clone MolmoBot

Follow the official MolmoBot setup instructions first.

Repository:

```text
https://github.com/allenai/MolmoBot
```

After cloning, check the MolmoSpaces dependency pin in:

```text
MolmoBot/MolmoBot/pyproject.toml
```

The released RB-Y1 MolmoBot setup expects MolmoSpaces around:

```text
allenai/molmospaces@cd23becebcf72dd93a4aa5872a60802d5eff03ef
```

### 2. Clone MolmoSpaces at the MolmoBot-pinned commit

Clone MolmoSpaces separately so you can patch it locally:

```bash
git clone https://github.com/allenai/molmospaces.git
cd molmospaces
git checkout cd23becebcf72dd93a4aa5872a60802d5eff03ef
```

This is important because latest MolmoSpaces `main` may not work directly with the released RB-Y1 MolmoBot checkpoint/eval code.

### 3. Use the RB-Y1 benchmark resources from `20260327`

When setting up RB-Y1 benchmark paths, use:

```text
molmospaces-bench-v2/20260327
```

not:

```text
molmospaces-bench-v2/20260325_1
```

and not:

```text
molmospaces-bench-v2/20260415
```

The expected RB-Y1 benchmark directories are:

```text
procthor-objaverse/rby1_benchmarks/pick_benchmark
procthor-objaverse/rby1_benchmarks/pnp_benchmark
ithor/rby1_benchmarks/opening_benchmark
procthor-10k/rby1_benchmarks/door_opening_benchmark
```

If your local MolmoSpaces code still points to `20260325_1`, update the RB-Y1 benchmark version/path logic according to the MolmoSpaces change associated with:

```text
e1a2fe16
```

### 4. Apply the RB-Y1 `root_body_id` type fix

File:

```text
molmo_spaces/robots/robot_views/rby1_view.py
```

Make sure the RB-Y1 gripper EE group stores an integer body id:

```python
root_body_id = model.body(f"{namespace}EE_BODY_{side[0].upper()}").id
```

Also make sure the RB-Y1 holo base group stores an integer body id:

```python
root_body_id = model.body(f"{namespace}base").id
```

This corresponds to PR #135 commit:

```text
8797f8818086174509d3a9c288eccbaf713ad532
```

### 5. Apply the RB-Y1 Pick contact-root comparison fix

File:

```text
molmo_spaces/tasks/pick_task.py
```

Before checking whether object contacts are robot contacts, map the RB-Y1 base body to the robot kinematic tree root:

```python
robot_root_body_id = data.model.body_rootid[
    int(self.env.current_robot.robot_view.base.root_body_id)
]
```

Then compare object contacts against:

```python
robot_root_body_id
```

instead of directly comparing against:

```python
self.env.current_robot.robot_view.base.root_body_id
```

This is needed because RB-Y1's base body is nested. MuJoCo contact roots are reported through `body_rootid`, so the comparison target also needs to be the kinematic tree root.

This corresponds to PR #135 commit:

```text
6ebeeb8a38a6899b583e98c5c59cb25ccab76efe
```

### 6. Apply the RB-Y1 Pick-and-Place contact-root comparison fix

File:

```text
molmo_spaces/tasks/pick_and_place_task.py
```

Apply the same kinematic-tree-root mapping for the robot-release contact check:

```python
robot_root_body_id = data.model.body_rootid[
    int(self._env.current_robot.robot_view.base.root_body_id)
]
```

Then compare robot-object contacts against:

```python
robot_root_body_id
```

This keeps the PnP success check consistent with RB-Y1's nested kinematic tree.

### 7. Install the patched MolmoSpaces into the same environment as MolmoBot

Use the same Python environment where you installed MolmoBot.

From the local MolmoSpaces checkout:

```bash
pip install -e .
```

Then install MolmoBot following the official MolmoBot instructions.

The important point is that MolmoBot and the patched MolmoSpaces should be installed in the same environment, so MolmoBot imports this patched MolmoSpaces copy rather than an incompatible unpatched version.

### 8. Verify the expected code is present

From the MolmoSpaces checkout, check that the RB-Y1 `.id` fixes are present:

```bash
rg 'root_body_id = model.body' molmo_spaces/robots/robot_views/rby1_view.py
```

The RB-Y1 gripper EE and holo base lines should include `.id`.

Also check that `pick_task.py` and `pick_and_place_task.py` use:

```python
data.model.body_rootid[
    int(...)
]
```

for the robot root contact comparison.

### 9. Optional sanity check: compile the touched files

This is not required to make the code run, but it is a useful quick check before launching long evaluations.

From the MolmoSpaces checkout:

```bash
python -m py_compile   molmo_spaces/robots/robot_views/rby1_view.py   molmo_spaces/tasks/pick_task.py   molmo_spaces/tasks/pick_and_place_task.py
```

### 10. Run a small RB-Y1 evaluation smoke test first

Before launching a large benchmark run, run one or a few RB-Y1 Pick/PnP episodes using the official MolmoBot evaluation entrypoint you are using.

The goal of the smoke test is to confirm:

1. MolmoBot imports the patched MolmoSpaces checkout.
2. RB-Y1 simulation starts normally.
3. The RB-Y1 benchmark paths resolve to `molmospaces-bench-v2/20260327`.
4. Pick/PnP evaluation completes without import or task errors.
5. Pick success is no longer artificially stuck at `0%` due to the RB-Y1 contact-check bug.

## Important Notes

1. Latest MolmoSpaces `main` may not work directly with the released RB-Y1 MolmoBot checkpoint/eval setup.
2. Start from the MolmoSpaces commit pinned by MolmoBot: `cd23becebcf72dd93a4aa5872a60802d5eff03ef`.
3. Use RB-Y1 benchmark resources from `molmospaces-bench-v2/20260327`.
4. Do not rely on `molmospaces-bench-v2/20260325_1` for RB-Y1 benchmarks, because it does not contain the needed `rby1_benchmarks/` directories.
5. Do not rely on `molmospaces-bench-v2/20260415` for public reproduction unless it is available in your dataset source.
6. Apply both PR #135 fixes, not only the `.id` fix.
7. The `.id` fix belongs in `molmo_spaces/robots/robot_views/rby1_view.py`.
8. The kinematic-tree-root comparison fix belongs in both `molmo_spaces/tasks/pick_task.py` and `molmo_spaces/tasks/pick_and_place_task.py`.
9. For RB-Y1, compare contacts against the robot kinematic tree root, not directly against the base move group body id.
10. Make sure the patched MolmoSpaces checkout is the one actually imported by MolmoBot in your Python environment.
