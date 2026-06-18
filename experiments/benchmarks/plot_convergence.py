"""Plot MVP convergence metrics.

Reads `experiments/mvp_sentiment/results/metrics.json` and writes a lightweight
SVG chart that can be opened directly in a browser or embedded in docs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "experiments" / "mvp_sentiment" / "results" / "metrics.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "benchmarks" / "convergence.svg"


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}. Run experiments/mvp_sentiment/run_mvp.py first."
        )

    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Metrics file is empty or invalid: {path}")
    return metrics


def normalize(values: Iterable[float], lower_is_better: bool = False) -> list[float]:
    values = list(values)
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        normalized = [0.5 for _ in values]
    else:
        normalized = [(value - min_value) / (max_value - min_value) for value in values]

    if lower_is_better:
        normalized = [1.0 - value for value in normalized]
    return normalized


def path_points(rounds: list[int], normalized: list[float], width: int, height: int) -> str:
    left = 80
    right = width - 40
    top = 40
    bottom = height - 70

    if len(rounds) == 1:
        x_values = [(left + right) / 2]
    else:
        x_values = [
            left + (idx / (len(rounds) - 1)) * (right - left)
            for idx in range(len(rounds))
        ]

    points = []
    for x_value, y_norm in zip(x_values, normalized):
        y_value = bottom - y_norm * (bottom - top)
        points.append(f"{x_value:.1f},{y_value:.1f}")
    return " ".join(points)


def render_svg(metrics: list[dict], output_path: Path) -> None:
    width = 920
    height = 520
    rounds = [int(item["round"]) for item in metrics]
    losses = [float(item["avg_loss"]) for item in metrics]
    accuracies = [float(item["accuracy"]) for item in metrics]
    epsilons = [float(item["epsilon_max"]) for item in metrics]

    loss_points = path_points(rounds, normalize(losses, lower_is_better=True), width, height)
    accuracy_points = path_points(rounds, normalize(accuracies), width, height)
    epsilon_points = path_points(rounds, normalize(epsilons), width, height)

    round_labels = []
    left = 80
    right = width - 40
    bottom = height - 70
    for idx, round_num in enumerate(rounds):
        x_value = (left + right) / 2 if len(rounds) == 1 else left + (idx / (len(rounds) - 1)) * (right - left)
        round_labels.append(
            f'<text x="{x_value:.1f}" y="{bottom + 30}" text-anchor="middle" '
            f'font-size="13" fill="#52525b">R{round_num}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="40" y="34" font-size="22" font-weight="700" fill="#18181b">FusionNet MVP Convergence</text>
  <text x="40" y="58" font-size="13" fill="#71717a">Local network simulation: weighted FedAvg across client updates</text>

  <line x1="80" y1="40" x2="80" y2="450" stroke="#d4d4d8" stroke-width="1"/>
  <line x1="80" y1="450" x2="880" y2="450" stroke="#d4d4d8" stroke-width="1"/>
  <line x1="80" y1="245" x2="880" y2="245" stroke="#e4e4e7" stroke-width="1" stroke-dasharray="4 5"/>

  <polyline fill="none" stroke="#0891b2" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{loss_points}"/>
  <polyline fill="none" stroke="#16a34a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{accuracy_points}"/>
  <polyline fill="none" stroke="#7c3aed" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="{epsilon_points}"/>

  <text x="96" y="92" font-size="13" fill="#0891b2">Loss improvement</text>
  <text x="96" y="116" font-size="13" fill="#16a34a">Accuracy</text>
  <text x="96" y="140" font-size="13" fill="#7c3aed">Epsilon spent</text>

  <circle cx="86" cy="87" r="5" fill="#0891b2"/>
  <circle cx="86" cy="111" r="5" fill="#16a34a"/>
  <circle cx="86" cy="135" r="5" fill="#7c3aed"/>

  {''.join(round_labels)}

  <text x="40" y="480" font-size="12" fill="#71717a">Final loss: {losses[-1]:.4f}</text>
  <text x="190" y="480" font-size="12" fill="#71717a">Final accuracy: {accuracies[-1] * 100:.2f}%</text>
  <text x="380" y="480" font-size="12" fill="#71717a">Final epsilon: {epsilons[-1]:.2f}</text>
</svg>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FusionNet MVP convergence metrics")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Path to metrics.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output SVG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    render_svg(metrics, args.output)
    print(f"Saved convergence chart to {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
