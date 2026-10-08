# Dataset results

`final/` links to `/bigtemp/xna8aw/nav_door/data/final`: the existing dataset
with navigation matched to door-opening speed and filtered to opened-with-grasp
trajectories. It contains 601 trajectories in two split views:

- `final/house/train` and `final/house/val`: preserved house split, 432/169.
- `final/trajectory/train` and `final/trajectory/val`: trajectory split, 511/90,
  stratified by push/pull with seed 42. Houses may appear in both splits.

Physical files live under `house/`; `trajectory/` links to those files. Pass
`final/house` or `final/trajectory` as the training data root. Each split needs
its own `valid_trajectory_index.json` before MolmoBot training.
Each trajectory has an H5 file and three camera videos.

`numbered/` and `quality_check/` inside the final dataset preserve clip IDs and
grasp selection metadata. The original, fixed-speed, and sample versions are
retired. The retained dataset was generated before the new baseline runner;
consolidation does not add point-marker videos or regenerate trajectories.

Future baseline runs, if requested, use `results/nav_door_point_matched/RUN_ID/`
and `/bigtemp/xna8aw/nav_door/baselines/nav_door_point_matched/RUN_ID/`, with raw,
recordings, dataset, resolved configuration, and stage manifest in each run.

Shared team resources are available through `../molmobot_molmospace/`: `dataset/` includes both split policies, and `media/` includes grasp-classified comparison MP4s.
