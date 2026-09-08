# Single-Panel Door Stability Evaluation

Date: 2026-09-14

## Slide 1 — Updated Evaluation Requirement

- Test whether RB-Y1 can reliably navigate to and push open a selected door.
- Use only physical single-panel doors; exclude both panels of every double
  doorway.
- Keep the selected target fixed throughout navigation, alignment, and opening.
- Evaluate the complete instruction as one task:
  `Navigate to the selected door and push it open`.

## Slide 2 — Single-Panel Dataset Filter

- MolmoSpaces scene metadata maps each articulated panel to its physical
  doorway, allowing classification by panel count.
- New generation now rejects any doorway whose panel count is not exactly one.
- The stopped campaign contains 126 eligible single-panel trajectories and 189
  excluded double-panel trajectories.
- Raw outputs remain preserved, while a generated eligibility manifest selects
  only single-panel H5 files for training.
- Future trajectories also save doorway, asset, panel-count, and single-panel
  metadata directly in H5.

## Slide 3 — Stability Experiments

- Matched `nav + open` and `open only` modes use the same cuRobo door-opening
  policy, isolating navigation and handoff failures.
- Each single-panel door receives 10 random-start attempts by default, without
  stopping early after successes.
- Available scopes test one door, a matched ten-door house set, or every
  single-panel door in houses 0–99; full evaluation retains A–F house shards.
- Push-only and pull-only branches reject starts from the wrong side and write
  to separate result directories.
- Per-door JSON manifests report attempts, successes, and success rates; both
  successful and failed trajectories are retained.
