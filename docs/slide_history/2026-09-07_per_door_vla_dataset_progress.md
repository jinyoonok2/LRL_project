# Per-Door RB-Y1 VLA Dataset Progress

Date: 2026-09-07

## Slide 1 — Today's Data-Generation Updates

- The sampler now checks every openable door panel, targeting three successful
  trajectories within 12 rollout attempts per target.
- The selected door is fixed before rollout: A* navigates to its handle and
  cuRobo opens the same door, using instructions such as
  `Navigate to the visible door and pull the door open`.
- Cross-house skipping and false irrecoverable-completion bugs were fixed.
- Every success is checkpointed immediately to H5/video, preventing data loss
  after a later crash.

## Slide 2 — Current Campaign

Snapshot: 2026-09-07 10:58 EDT

- A: complete, 44 successes; B: complete, 68; D: complete, 71.
- C stopped early with 12 successes; houses 28–41 need a targeted rerun.
- E is processing house 72 with 43 successes so far; F remains queued.
- Corrected campaign total so far: 238 saved successful trajectories.

## Slide 3 — Training Readiness and Next Steps

- Outputs already use the original MolmoBot-compatible H5 format; the training
  loop itself remains unchanged.
- Finish E/F, rerun missing C houses, group duplicate panels by physical
  doorway, and create house-separated train/validation splits.
- Validate trajectories, build `valid_trajectory_index.json`, calculate
  statistics, then run a multi-GPU fine-tuning smoke test.
- To prevent learned mid-rollout target switching, condition every step on the
  same marked initial goal image during both training and inference.
