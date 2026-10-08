# Navigation-to-Door Dataset and MolmoBot Training Progress

Date: 2026-10-05 (America/New_York)

## Slide 1 Molmobot: Progress Since the Previous Update

Finalized 601 navigation-to-door demonstrations, filtered for successful opening with grasp and rerendered so navigation follows the recorded opening-speed profile.

Organized house and trajectory train/validation splits, and consolidated future generation into modular extension classes.

Completed LLM + action-expert fine-tuning on the trajectory split. Full evaluation of 90 held-out trajectory starts completed with 28 successes (31.1%) and no process errors.

## Slide 2 Molmobot: How Navigation Speed Was Matched

For each trajectory, measure median door-handle travel and hinge-angle change per frame during the recorded opening phase.

Use those measurements as limits for base translation and rotation during navigation, final alignment, and door approach. Subdivide larger movements into smaller interpolated steps while preserving the path.

Regenerate the H5 trajectory and all three camera videos at 10 fps. Additional frames slow navigation; opening keeps its original timing.

Matching caps navigation steps using typical opening motion, rather than enforcing identical speeds or durations. Minimum limits prevent excessive subdivision when opening motion is nearly stationary.

## Slide 3 Molmobot: Ground Truth and Model Comparisons

Each comparison pairs the recorded expert demonstration on the left with model inference from the corresponding validation start on the right.

Success example: episode 001, house 1.

[Successful model comparison](media/success/opened_with_grasp/episode_001_house_1_success_opened_with_grasp_comparison.mp4)

[Success comparison video (MP4)](media/success/opened_with_grasp/episode_001_house_1_success_opened_with_grasp_comparison.mp4)

Failure example: episode 000, house 1.

[Failed model comparison](media/failure/episode_000_house_1_failure_comparison.mp4)

[Failure comparison video (MP4)](media/failure/episode_000_house_1_failure_comparison.mp4)

[All 28 success comparisons](media/success/README.md) and [all 62 failure comparisons](media/failure/README.md). Playback remains at 1x; the shorter video holds its final frame.

## Slide 4 Molmobot: Final Dataset and Split Organization

Retained 601 trajectories with H5 records and head/left-wrist/right-wrist videos. The full rerender processed 591 trajectories and retained 10 existing outputs; superseded datasets were removed.

| Split policy | Train | Validation | Purpose |
|---|---:|---:|---|
| House | 432 | 169 | Held-out-house generalization |
| Trajectory | 511 | 90 | Held-out trajectories; houses may overlap |

Physical files live under `final/house`; `final/trajectory` links to them. The trajectory split uses seed 42, approximately 15% validation, and push/pull stratification.

`numbered/` and `quality_check/opened_with_grasp/` reference the same head videos. Quality indexes record grasp classifications and metrics.

## Slide 5 Molmobot: Modular Framework and Verified Training Inputs

One YAML-driven baseline coordinates generation, split organization, speed matching, point prompts, and validation. Separate classes own navigation, rollout scheduling, rerendering, grasp checks, and splitting.

New workflow changes live in extensions; previously accepted upstream compatibility fixes remain pinned. Training used existing demonstrations without rerunning the baseline or adding its new point overlays.

October 2 training jobs failed because trajectory indexes were missing. The preparation class now builds those indexes, adds wrist-video aliases, and checks video/action alignment without changing H5 files.

Verified 511 train / 90 validation trajectories, 433,383 observation frames, and 1,202 wrist-video aliases; training then completed successfully.

## Slide 6 Molmobot: Training and Full Evaluation Completed

| Stage | GPU | Status at this update |
|---|---|---|
| LLM + action-expert training | H100 94GB | Completed; job 6668431 |
| Full navigation-to-door evaluation | A40/A100/H100 eligible | All 90 completed; array 6687763 |

Training completed 5,000 steps in 3 h 46 min 21 s on the trajectory split, with vision backbone and connector frozen. Final checkpoint: `/bigtemp/xna8aw/nav_door/train/final/trajectory/llm/step5000`.

Full evaluation used all 90 held-out trajectory starts, seed 42, and a per-episode horizon of at least 2,000 steps or twice the recorded length. Inputs match training: no point prompts, extra conditioning image, or camera warping.

Full-task success was 28/90 (31.1%); the other 62 episodes failed the task, with no process errors. Ground-truth/model comparisons are organized under `media/success` and `media/failure`. A pretrained-model comparison on the same starts is still needed to establish improvement.

## Evidence and References

[LLM training log](../../slurm_logs/molmobot-final-trajectory-llm-6668431.out): final step, checkpoint save, and training completion.

[Dataset preparation report](../../results/final/trajectory/training_preparation.json): indexed counts, aliases, and frame total.

Training job 6668431 completed; the earlier door-only evaluation 6676831 and full validation array 6687763 have also completed.

[Extension framework](../../lrl_molmospaces_extensions/README.md), [project environment](../PROJECT_ENVIRONMENT.md), [setup and compatibility](../SETUP.md), and [research plan](../RESEARCH_PLAN.md).

## Follow-up on 2026-10-06

The initial standard door-opening benchmark completed with 2/10 successes (20%) and no process errors. This was separate from the dataset validation split.

Full closed-loop navigation-to-door evaluation covered all 90 held-out trajectory starts (array 6687763): 28 successes, 62 task failures, and no process errors. These are trajectory-held-out results, not held-out-house generalization.

[Full validation comparison GIFs](media/README.md): all 90 episodes, paired with their matching expert recordings.

## Comparison media update on 2026-10-07

Replaced all earlier media with 90 matched ground-truth/model comparisons: 28 successful openings and 62 failures. MP4s are retained; GIF versions have been removed. Each filename and image includes the model outcome. These results use the trajectory validation split; they do not establish held-out-house generalization.

## Grasp-quality update on 2026-10-07

Applying the same 30-frame recorded-grasp threshold as the dataset gives 1 opened-with-grasp episode (1.1% of 90), 27 opened-without-qualifying-grasp episodes, and 62 unsuccessful openings. Of the 27 below-threshold successful openings, 26 recorded zero grasp frames and one recorded a single frame. Raw opening success (31.1%) should therefore be distinguished from grasp-qualified success (1.1%).

[With qualifying grasp](media/success/opened_with_grasp/README.md), [without qualifying grasp](media/success/opened_without_grasp/README.md), and [failed openings](media/failure/README.md). All comparison videos are MP4s.
