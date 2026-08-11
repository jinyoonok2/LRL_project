#!/usr/bin/env python3
"""Submit MolmoBot training in checkpoint-validated fragments.

This script does not change MolmoBot's trainer. It wraps the authors' training
entrypoint and submits only the next fragment whose prerequisites are satisfied.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def load_plan(path: Path) -> dict[str, Any]:
    with path.open() as f:
        plan = json.load(f)
    if "fragments" not in plan or not isinstance(plan["fragments"], list):
        raise ValueError("Plan must contain a 'fragments' list.")
    return plan


def as_path(value: str | Path, project_root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and project_root is not None:
        path = project_root / path
    return path


def checkpoint_valid(path: Path, *, require_optimizer: bool = True) -> bool:
    """Return True when a MolmoBot sharded training checkpoint looks complete."""
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.yaml").exists():
        return False
    if require_optimizer:
        if not (path / "model_and_optim" / ".metadata").exists():
            return False
        if not list((path / "train").glob("rank*.pt")):
            return False
    return True


def latest_checkpoint(save_folder: Path) -> Path | None:
    checkpoints = save_folder / "checkpoints"
    if not checkpoints.exists():
        return None
    steps: list[tuple[int, Path]] = []
    for child in checkpoints.iterdir():
        if child.is_dir() and child.name.startswith("step"):
            try:
                steps.append((int(child.name[4:]), child))
            except ValueError:
                continue
    return max(steps, default=(0, None))[1]


def fragment_checkpoint(fragment: dict[str, Any], defaults: dict[str, Any], project_root: Path) -> Path:
    save_folder = as_path(fragment.get("save_folder", defaults["save_folder"]), project_root)
    target_step = int(fragment["target_step"])
    return save_folder / "checkpoints" / f"step{target_step}"


def resolved_load_path(
    index: int,
    fragments: list[dict[str, Any]],
    defaults: dict[str, Any],
    project_root: Path,
) -> Path | None:
    fragment = fragments[index]
    raw_load_path = fragment.get("load_path")
    if raw_load_path == "previous":
        if index == 0:
            raise ValueError("First fragment cannot use load_path='previous'.")
        return fragment_checkpoint(fragments[index - 1], defaults, project_root)
    if raw_load_path:
        return as_path(raw_load_path, project_root)
    if index > 0 and fragment.get("require_previous", True):
        return fragment_checkpoint(fragments[index - 1], defaults, project_root)
    return None


def first_pending_fragment(plan: dict[str, Any], project_root: Path) -> int | None:
    defaults = plan.get("defaults", {})
    fragments = plan["fragments"]
    for i, fragment in enumerate(fragments):
        expected = fragment_checkpoint(fragment, defaults, project_root)
        if checkpoint_valid(expected):
            continue
        load_path = resolved_load_path(i, fragments, defaults, project_root)
        if load_path is not None and not checkpoint_valid(load_path):
            raise RuntimeError(
                f"Refusing to submit fragment {i} ({fragment.get('name', 'unnamed')}): "
                f"required checkpoint is missing or incomplete: {load_path}"
            )
        return i
    return None


def quote_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def timing_line(event: str) -> str:
    return f'echo "[chunked-train] {event} iso=$(date -Is) epoch=$(date +%s)"'


def train_command(
    fragment: dict[str, Any],
    defaults: dict[str, Any],
    project_root: Path,
    load_path: Path | None,
) -> list[str]:
    checkpoint = as_path(fragment.get("checkpoint", defaults["base_checkpoint"]), project_root)
    save_folder = as_path(fragment.get("save_folder", defaults["save_folder"]), project_root)
    data_paths = fragment.get("data_paths", defaults.get("data_paths"))
    if not data_paths:
        raise ValueError("Each fragment needs data_paths, either directly or in defaults.")

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc-per-node={fragment.get('num_gpus', defaults.get('num_gpus', 1))}",
        f"--master_port={fragment.get('master_port', defaults.get('master_port', 29501))}",
        "launch_scripts/train_molmobot.py",
        str(checkpoint),
        "--data_paths",
        *[str(as_path(p, project_root)) for p in data_paths],
        "--seq_len",
        str(fragment.get("seq_len", defaults["seq_len"])),
        "--device_batch_size",
        str(fragment.get("device_batch_size", defaults["device_batch_size"])),
        "--global_batch_size",
        str(fragment.get("global_batch_size", defaults["global_batch_size"])),
        "--action_preset",
        str(fragment.get("action_preset", defaults["action_preset"])),
        "--camera_preset",
        str(fragment.get("camera_preset", defaults["camera_preset"])),
        f"--save_folder={save_folder}",
        f"--max_duration={int(fragment['target_step'])}",
        f"--stop_at={int(fragment['target_step'])}",
        f"--save_interval={fragment.get('save_interval', defaults.get('save_interval', 1000))}",
        f"--save_num_checkpoints_to_keep={fragment.get('save_num_checkpoints_to_keep', defaults.get('save_num_checkpoints_to_keep', 2))}",
    ]

    stats_path = fragment.get("stats_path", defaults.get("stats_path"))
    if stats_path:
        cmd.append(f"--stats_path={as_path(stats_path, project_root)}")
    if load_path is not None:
        cmd.append(f"--load_path={load_path}")
    if fragment.get("no_val", defaults.get("no_val", False)):
        cmd.append("--no_val")

    ephemeral = fragment.get("save_interval_ephemeral", defaults.get("save_interval_ephemeral"))
    if ephemeral is not None:
        cmd.append(f"--save_interval_ephemeral={ephemeral}")

    retention = fragment.get("checkpoint_retention_frequency", defaults.get("checkpoint_retention_frequency"))
    if retention is not None:
        cmd.append(f"--checkpoint_retention_frequency={retention}")

    for arg in defaults.get("extra_args", []) + fragment.get("extra_args", []):
        cmd.append(str(arg))
    return cmd


def sbatch_command(plan: dict[str, Any], fragment: dict[str, Any], train_cmd: list[str], project_root: Path) -> list[str]:
    defaults = plan.get("defaults", {})
    slurm = defaults.get("slurm", {}) | fragment.get("slurm", {})
    log_dir = as_path(slurm.get("log_dir", "slurm_logs"), project_root)
    log_dir.mkdir(parents=True, exist_ok=True)

    molmobot_dir = as_path(defaults.get("molmobot_dir", "MolmoBot/MolmoBot"), project_root)
    modules = slurm.get("modules", [])
    exports = defaults.get("env", {}) | fragment.get("env", {})

    setup_lines = ["set -eo pipefail", f"cd {shlex.quote(str(molmobot_dir))}"]
    setup_lines.insert(0, "JOB_START_TS=$(date +%s)")
    setup_lines.insert(1, timing_line("job_start"))
    setup_lines.append("SETUP_START_TS=$(date +%s)")
    setup_lines.append(timing_line("setup_start"))
    if modules:
        setup_lines.append("module load " + " ".join(shlex.quote(str(m)) for m in modules))
    conda_activate = slurm.get("conda_activate")
    if conda_activate:
        setup_lines.append('source "$(conda info --base)/etc/profile.d/conda.sh"')
        setup_lines.append("conda activate " + shlex.quote(str(conda_activate)))
    setup_lines.append("export PYTHONPATH=${PYTHONPATH:-.}")
    for key, value in exports.items():
        setup_lines.append(f"export {key}={shlex.quote(str(value))}")
    setup_lines.append("SETUP_END_TS=$(date +%s)")
    setup_lines.append('echo "[chunked-train] setup_end iso=$(date -Is) epoch=${SETUP_END_TS} duration_sec=$((SETUP_END_TS - SETUP_START_TS))"')
    setup_lines.append("TRAIN_START_TS=$(date +%s)")
    setup_lines.append(timing_line("train_command_start"))
    setup_lines.append("set +e")
    setup_lines.append(quote_join(train_cmd))
    setup_lines.append("TRAIN_STATUS=$?")
    setup_lines.append("set -e")
    setup_lines.append("TRAIN_END_TS=$(date +%s)")
    setup_lines.append('echo "[chunked-train] train_command_end iso=$(date -Is) epoch=${TRAIN_END_TS} duration_sec=$((TRAIN_END_TS - TRAIN_START_TS)) exit_code=${TRAIN_STATUS}"')
    setup_lines.append("JOB_END_TS=$(date +%s)")
    setup_lines.append('echo "[chunked-train] job_end iso=$(date -Is) epoch=${JOB_END_TS} duration_sec=$((JOB_END_TS - JOB_START_TS)) exit_code=${TRAIN_STATUS}"')
    setup_lines.append("exit ${TRAIN_STATUS}")
    wrapped = "bash -lc " + shlex.quote("\n".join(setup_lines))

    name = fragment.get("name", f"molmobot-step{fragment['target_step']}")
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--output={log_dir / (name + '-%j.out')}",
        f"--error={log_dir / (name + '-%j.err')}",
        f"--partition={slurm.get('partition', 'gpu')}",
        f"--gres=gpu:{fragment.get('num_gpus', defaults.get('num_gpus', 1))}",
        f"--cpus-per-task={slurm.get('cpus_per_task', 10)}",
        f"--mem={slurm.get('mem', '100G')}",
        f"--time={slurm.get('time', '12:00:00')}",
    ]
    if slurm.get("account"):
        cmd.append(f"--account={slurm['account']}")
    if slurm.get("constraint"):
        cmd.append(f"--constraint={slurm['constraint']}")
    cmd.append(f"--wrap={wrapped}")
    return cmd


def submit_next(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).expanduser().resolve()
    plan = load_plan(plan_path)
    project_root = Path(plan.get("defaults", {}).get("project_root", ".")).expanduser().resolve()
    index = first_pending_fragment(plan, project_root)
    if index is None:
        print("All fragments already have valid checkpoints.")
        return

    defaults = plan.get("defaults", {})
    fragment = plan["fragments"][index]
    load_path = resolved_load_path(index, plan["fragments"], defaults, project_root)
    train_cmd = train_command(fragment, defaults, project_root, load_path)
    sbatch = sbatch_command(plan, fragment, train_cmd, project_root)

    print(f"Next fragment: {index} ({fragment.get('name', 'unnamed')})")
    if load_path:
        print(f"Resuming from: {load_path}")
    print("Training command:")
    print(quote_join(train_cmd))
    print("Slurm command:")
    print(quote_join(sbatch))
    if args.dry_run:
        return
    result = subprocess.run(sbatch, check=True, text=True, capture_output=True)
    print(f"Submitted Slurm job: {result.stdout.strip()}")


def check(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).expanduser().resolve()
    plan = load_plan(plan_path)
    project_root = Path(plan.get("defaults", {}).get("project_root", ".")).expanduser().resolve()
    for i, fragment in enumerate(plan["fragments"]):
        expected = fragment_checkpoint(fragment, plan.get("defaults", {}), project_root)
        status = "complete" if checkpoint_valid(expected) else "pending"
        print(f"{i}: {fragment.get('name', 'unnamed')} -> {status} ({expected})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit-next", help="Submit the first fragment whose checkpoint is missing.")
    submit.add_argument("--plan", required=True, help="Path to chunked training JSON plan.")
    submit.add_argument("--dry-run", action="store_true", help="Print commands without submitting to Slurm.")
    submit.set_defaults(func=submit_next)

    status = sub.add_parser("status", help="Print checkpoint status for every fragment.")
    status.add_argument("--plan", required=True, help="Path to chunked training JSON plan.")
    status.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
