"""Summarize sequential UA/RA results as a CSV."""
import argparse
import csv
from pathlib import Path

import torch


REQUESTS = ["Abstractionism", "Byzantine", "Cartoon", "Cold_Warm", "Ukiyoe", "Van_Gogh"]
METHODS = ["esd", "salun", "ediff"]


def load_accuracy(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing accuracy result: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    return {name: float(value) for name, value in payload["acc"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("eval_results/sequential"),
        help="Root containing per-method, per-step evaluation outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/sequential/table_a5_reproduction.csv"),
        help="CSV path for the sequential UA/RA table.",
    )
    args = parser.parse_args()

    rows = []
    for method in METHODS:
        style_by_step = []
        object_by_step = []
        for step in range(1, 7):
            step_dir = args.eval_root / method / f"step_{step:02d}" / "accuracy"
            style_by_step.append(load_accuracy(step_dir / "style" / "result.pth"))
            object_by_step.append(load_accuracy(step_dir / "object" / "result.pth"))

        for target_idx, target in enumerate(REQUESTS):
            values = [""] * 6
            for step_idx in range(target_idx, 6):
                values[step_idx] = round((1.0 - style_by_step[step_idx][target]) * 100.0, 2)
            rows.append([method, "UA", target, *values])

        ra_values = []
        for step_idx in range(6):
            retained_style_accs = [
                acc for name, acc in style_by_step[step_idx].items() if name not in REQUESTS
            ]
            object_accs = list(object_by_step[step_idx].values())
            if not retained_style_accs or not object_accs:
                raise ValueError(f"Empty accuracy results: {method}, step {step_idx + 1}")
            style_ra = sum(retained_style_accs) / len(retained_style_accs)
            object_ra = sum(object_accs) / len(object_accs)
            ra_values.append(round((style_ra + object_ra) * 50.0, 2))
        rows.append([method, "RA", "all retained concepts", *ra_values])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "request", *[f"T{n}" for n in range(1, 7)]])
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
