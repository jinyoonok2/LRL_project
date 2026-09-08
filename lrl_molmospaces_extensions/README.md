# LRL MolmoSpaces Extensions

This package contains the long-horizon VLA tasks, policies, navigation changes,
data-generation configs, and per-door stability accounting. The pinned
`molmospaces/` checkout remains the upstream implementation plus the separately
accepted pre-VLA compatibility fixes.

Launch a fixed-attempt door evaluation with:

```bash
scripts/run_rby1_door_stability_kuorobot01.sh open-only shard-a push
```

The optional fourth argument changes the attempt count for smoke tests:

```bash
scripts/run_rby1_door_stability_kuorobot01.sh open-only same-door push 2
```

Validate a completed run with:

```bash
python -m lrl_molmospaces_extensions.datagen.validate_shard RUN_DIR \
  --expected-attempts 10
```
