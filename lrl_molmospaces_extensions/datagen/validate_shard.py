"""Validate fixed-attempt per-door stability output."""

import argparse
import json
from pathlib import Path


class ShardValidator:
    def __init__(self, run_dir: Path, expected_attempts: int = 10):
        self.run_dir = run_dir
        self.expected_attempts = expected_attempts

    def validate(self) -> tuple[bool, dict]:
        manifests = sorted((self.run_dir / "per_door_manifests").glob("house_*.json"))
        exclusions = sorted((self.run_dir / "house_exclusions").glob("house_*.json"))
        failures = []
        door_count = 0
        successes = 0
        attempts = 0

        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text())
            for door_name, result in manifest.get("doors", {}).items():
                door_count += 1
                attempts += int(result["attempts"])
                successes += int(result["successes"])
                if (
                    result["status"] != "complete"
                    or int(result["attempts"]) != self.expected_attempts
                ):
                    failures.append(
                        {
                            "house_index": manifest["house_index"],
                            "door_body_name": door_name,
                            "status": result["status"],
                            "attempts": result["attempts"],
                            "placement_failures": result.get("placement_failures", 0),
                        }
                    )

        summary = {
            "run_dir": str(self.run_dir),
            "expected_attempts_per_door": self.expected_attempts,
            "validated_doors": door_count,
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts else None,
            "excluded_houses": [json.loads(path.read_text()) for path in exclusions],
            "incomplete_doors": failures,
        }
        output_path = self.run_dir / "validation_summary.json"
        output_path.write_text(json.dumps(summary, indent=2))
        return bool(manifests) and not failures, summary


def validate_run(run_dir: Path, expected_attempts: int) -> tuple[bool, dict]:
    return ShardValidator(run_dir, expected_attempts).validate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-attempts", type=int, default=10)
    args = parser.parse_args()
    valid, summary = validate_run(args.run_dir, args.expected_attempts)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
