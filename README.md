# Live Robotics Lab — RB-Y1 MolmoBot / MolmoSpaces

This project brings up and evaluates the released RB-Y1 MolmoBot checkpoint,
and generates long-horizon navigation-to-door-opening demonstrations for
fine-tuning. Custom generation, quality checking, dataset splitting, and
rerendering live in `lrl_molmospaces_extensions/`.

Start with these maintained guides:

- [Project environment, storage, and repository state](docs/PROJECT_ENVIRONMENT.md)
- [Setup, pinned commits, and compatibility fixes](docs/SETUP.md)
- [MolmoBot fork changes and reproduction](MolmoBot/FORK_CHANGES.md)
- [MolmoSpaces fork changes and reproduction](molmospaces/FORK_CHANGES.md)
- [Research plan and next steps](docs/RESEARCH_PLAN.md)
- [Extension commands](lrl_molmospaces_extensions/README.md)

The current branch is `rby1-long-horizon-vla-training`. After the 2026-10-08
cleanup, the other local branches are `main` and `rby1`. See the environment
guide for exact submodule commits and the cleanup record.

## Repository layout

- `MolmoBot/`: model, training, and evaluation submodule.
- `molmospaces/`: simulation, assets, benchmark tasks, and planner submodule.
- `lrl_molmospaces_extensions/`: custom long-horizon data-generation pipeline.
- `configs/` and `scripts/`: experiment configuration and launchers.
- `results/`: working final dataset and future baseline run shortcuts.
- `molmobot_molmospace/`: shared project link, with `dataset/` splits and `media/` comparison videos.
- `docs/`: current guides and historical research plans.
- `docs/slide_history/`: dated experiment and presentation snapshots.
- `slurm_logs/`: scheduler outputs.

## Evaluation

From this directory with the `mlspaces` environment active:

```bash
python scripts/rby1_runner.py describe --config configs/rby1_eval.yaml
python scripts/rby1_runner.py eval --config configs/rby1_eval.yaml --tasks pick --indices 10 --dry-run
```

Check the benchmark-version discrepancy documented in the setup guide before
submitting evaluations. A completed process does not establish task success;
inspect episode results, H5 trajectories, and videos.

Historical benchmark evaluations ran successfully after compatibility fixes,
but Pick/PnP remained substantially below the published results. The June
summary recorded `pick 0/30`, `pnp 1/30`, `opening 9/30`, and
`door_opening 15/30`; these are historical results, not current measurements.

The dated plans and slides explain how the research evolved. Their job status,
file paths, and proposed next steps should be read as historical snapshots.
