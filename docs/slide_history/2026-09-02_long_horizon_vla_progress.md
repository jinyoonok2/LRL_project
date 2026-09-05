# Long-Horizon RB-Y1 VLA Progress

Date: 2026-09-02

## Slide 1 — Goal and Research Question

- We are generating continuous RB-Y1 demonstrations for the long-horizon task `navigate → approach target door → open door`, which is not represented as one combined task in the released MolmoBot training data.
- Our main research question is whether `MolmoBot-RBY1Multitask` can be fine-tuned on custom long-horizon trajectories and later generalized to other custom navigation-to-manipulation datasets.
- The immediate objective is therefore not only to make the planner pipeline succeed, but to produce demonstrations whose target, observations, instructions, robot states, and actions provide enough information for a VLA to learn the intended behavior.

## Slide 2 — Expert Data-Generation Pipeline

- MolmoSpaces creates the scene, RB-Y1 robot, cameras, target door, and task; A* generates navigation actions, a custom policy handles final alignment and phase switching, and cuRobo generates door-opening actions.
- Successful episodes are saved as continuous H5 trajectories containing RGB observations, robot state, language instruction, action chunks, policy phases, target metadata, and corresponding videos; these planner-generated demonstrations become the ground-truth examples for imitation fine-tuning.
- We initially tested `navigation → pick-and-place`, but unreliable final approach and grasp planning led us to use `navigation → door opening` as the first long-horizon learning task.

## Slide 3 — Reliability Improvements and Results

- We improved navigation execution by reducing excessive waypoint density, adding intermediate waypoint tolerance, preventing the short-path `m > k must hold` spline failure in a custom nav-door planner, and aligning the base before manipulation.
- Balanced handoff around `1.10–1.25m` performed better than strict `0.85–0.95m` alignment: observed medium-distance completion improved from 2/11 to 2/8, while long-distance completion improved from 1/11 to 2/9.
- Connectivity-aware sampling now rejects start poses disconnected from the target on the A* occupancy graph; after this change, medium `2–4m` achieved 4/9 full-task successes and long `4–8m` achieved 4/8, while all 17 accepted episodes completed navigation and reached an aligned handoff.
- Navigation feasibility is now substantially improved; the main remaining planner bottleneck is cuRobo door-opening execution after a valid handoff.

## Slide 4 — Simplified Fine-Tuning Task

- Following project feedback, the first VLA experiment will avoid room graphs, hidden targets, and multi-door search; RB-Y1 will start directly in front of one clearly visible target door at a close-to-medium distance of `0.8–3.0m`.
- The sampler will retain only connected start poses where the selected door is clearly visible in the initial head-camera image and no other door is substantially competing with it, producing the instruction “Navigate to the visible door and open it.”
- This simple setup isolates the main research question—whether MolmoBot can learn a continuous navigation-to-opening sequence—without requiring semantic mapping, route planning, memory, or target-search behavior at the same time.
- Point-prompt grounding remains available as an optional experiment, but it is not part of the primary fine-tuning dataset.

## Slide 5 — Training Readiness

- Eight successful connectivity-aware trajectories were organized into a leakage-aware split with six training trajectories from houses 1/4/7 and two validation trajectories from held-out house 22.
- All H5 files and videos passed validation, both `valid_trajectory_index.json` files were generated, and the MolmoBot loader successfully produced three RGB inputs, a 22-dimensional robot state, a `16 × 20` action chunk, and the long-horizon instruction.
- A real single-GPU smoke test loaded the five-billion-parameter model, released checkpoint, optimizer, normalization statistics, and dataloader, but the first forward pass exceeded the RTX 4090’s 24GB memory.
- This proves structural data compatibility, but no optimizer step or behavioral MolmoBot evaluation has completed yet; the eight trajectories are sufficient for infrastructure testing, not meaningful long-horizon learning.

## Slide 6 — Current Development and Next Steps

- The new primary configuration generates successful trajectories from one clearly visible door at `0.8–3.0m`, while retaining the proven connectivity filter, balanced handoff, final base orientation, and planner reliability fixes.
- The next step is to run this simplified configuration, review the initial videos, and generate a larger set of successful trajectories across multiple houses while preserving a house-separated validation split.
- After repairing video references and validating/indexing the new H5 files, we will run a multi-GPU MolmoBot training smoke test and then fine-tune `MolmoBot-RBY1Multitask` on the larger visible-door dataset.
- Final evaluation will place the model in the same simple task structure on held-out houses and test whether MolmoBot alone can perform `navigate → approach → open`, without A* or cuRobo during inference.
