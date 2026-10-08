# Long-Horizon RB-Y1 Research Plan

Consolidated: 2026-10-03 from the earlier navigation-to-manipulation plan, door-opening roadmap, and dated research updates. Proposed experiments below are not evidence of completed runs. Check artifacts and logs before updating their status.

## Goal

Fine-tune the released `MolmoBot-RBY1Multitask` checkpoint on our own continuous mobile-manipulation demonstrations. Test whether the learned policy can navigate, approach a target, and manipulate it using camera observations, robot state, and language. Expert A*/cuRobo policies generate training demonstrations; final model evaluation should use MolmoBot to choose actions.

## Initial task scope

Focus on navigation followed by door opening. The earlier navigation-to-pick-and-place prototype remains a possible later task, but approach and grasp reliability made door opening the first learning target.

- Use one clearly visible single-panel door and keep the selected target fixed throughout the episode.
- Start with close-to-medium distances (`0.8–3.0m` in the earlier visible-door plan).
- Require connected navigation starts and reject substantially competing door views.
- Keep push and pull behavior explicit; isolate push opening when studying stability.
- Exclude room-graph routing, hidden-target search, and dynamic target switching from the initial experiment. Point prompts or a fixed marked goal image are optional grounding experiments, not assumed model inputs.

## Existing foundation

The extension package provides composite tasks and samplers, A* navigation, final base alignment, cuRobo handoff, per-door accounting, and immediate successful-trajectory checkpointing. Later utilities add grasp-quality checks, train/validation splits, and navigation-speed rerendering.

Historical small sweeps favored balanced handoff over strict distance limits. Connectivity-aware sampling reduced impossible starts; cuRobo execution remained a bottleneck. These findings motivated the current pipeline, but the small samples do not establish general success rates. Dated results are preserved in [slide history](slide_history/README.md).

## Next steps

1. **Review demonstration quality.** Audit target consistency, single-panel eligibility, actual grasp, navigation-to-opening continuity, and H5/video alignment. Opening a door without grasping is a separate quality outcome. Record rejection reasons and preserve source data.
2. **Finalize the dataset.** Select usable opened-with-grasp demonstrations and compare recorded navigation with the opening-matched speed variant. Avoid accidental duplication between variants. Use house-separated validation for generalization; identify trajectory splits explicitly when houses overlap.
3. **Verify training inputs.** Validate H5/video references, generate `valid_trajectory_index.json`, calculate action/state statistics, and inspect a MolmoBot loader batch. Confirm camera/state/action ordering and that rerendered observations remain aligned with actions.
4. **Run a small fine-tuning experiment.** Start from the released checkpoint using the existing training stack. Verify optimizer steps, checkpoint saving, and resume behavior before scaling. The historical 24GB GPU smoke test ran out of memory on its first forward pass; choose resources from the current launcher and model requirements.
5. **Evaluate the learned policy.** Test held-out houses with MolmoBot choosing actions. Report full-task success and navigation, handoff, grasp, and opening failures separately. Keep planner and recorded-replay runs as controls, and use seeded model action sampling for repeat comparisons.
6. **Scale after evidence.** Expand houses and demonstrations once data loading, training, and behavioral evaluation work. Monitor whether navigation improves while manipulation degrades. Consider navigation-to-pick/place and longer distances afterward.

## Planner experiments if rollout yield remains low

- Compare matched `nav-open` and `open-only` runs on the same doors to separate navigation/handoff failures from manipulation failures.
- Tune handoff distance, yaw tolerance, final-alignment target and step budget using recorded phase metadata.
- Check handle reachability and correct push/pull side before expensive planning; confirm what the current implementation already checks before adding behavior.
- Explore diverse pregrasp offsets, approach angles, and standoff poses instead of repeating an identical failed plan.
- Revisit medium/long distance curricula after the visible-door baseline is established.

Earlier production plans targeted three successes within twelve attempts per door. Later stability experiments use fixed executed attempts, ten by default. Choose the protocol according to the experiment and record it; these are different sampling objectives.

## Main research risks

Navigation endpoints may be valid for A* but unsuitable for manipulation. Long episodes and changed navigation speeds may alter the model's training distribution. Weak target grounding can produce target switching. Narrow fine-tuning can reduce existing manipulation ability. Successful expert demonstrations alone do not establish learned-policy success.

Environment paths, source commits, and compatibility patches belong in [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md) and [SETUP.md](SETUP.md). Operational extension commands belong in the [extensions README](../lrl_molmospaces_extensions/README.md).
