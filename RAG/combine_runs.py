from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DEFAULT_OUTPUT = RUNS_DIR / "combined_runs.json"


def load_run_files(runs_dir: Path) -> list[dict]:
    """Load every per-ticket JSON run file in the given directory."""
    runs: list[dict] = []

    for run_file in sorted(runs_dir.glob("TKT-*.json")):
        with run_file.open(encoding="utf-8") as handle:
            runs.append(json.load(handle))

    return runs


def combine_runs(runs_dir: Path, output_file: Path) -> Path:
    """Combine all ticket run files into a single JSON document."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    runs = load_run_files(runs_dir)
    payload = {
        "agent": "rag",
        "source_directory": str(runs_dir),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "count": len(runs),
        "runs": runs,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine RAG run JSON files into one file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the combined JSON file (default: runs/combined_runs.json)",
    )
    args = parser.parse_args()

    output_file = combine_runs(RUNS_DIR, args.output)
    print(f"Wrote combined RAG runs to {output_file}")


if __name__ == "__main__":
    main()