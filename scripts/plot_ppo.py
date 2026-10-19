#!/usr/bin/env python3
"""Plot the PPO training report written by wired-train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ppo.json")
    output = Path(sys.argv[2] if len(sys.argv) > 2 else source.with_suffix(".png"))
    if not source.is_file():
        print(f"missing report: {source}", file=sys.stderr)
        return 1

    report = json.loads(source.read_text(encoding="utf-8"))
    rows = report["updates"]
    updates = [row["update"] for row in rows]
    reward = [row["mean_reward"] for row in rows]
    ball_y = [row["mean_ball_y_velocity"] for row in rows]
    policy_loss = [row["policy_loss"] for row in rows]
    value_loss = [row["value_loss"] for row in rows]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    figure.suptitle("wired-to-walk / PPO baseline", fontsize=13)

    axes[0].plot(updates, reward, color="#f08a3c", linewidth=1.4, label="rollout reward")
    axes[0].axhline(
        report["initial_eval"]["mean_reward"],
        color="#8f8f8f",
        linestyle="--",
        linewidth=1,
        label="initial deterministic eval",
    )
    axes[0].axhline(
        report["final_eval"]["mean_reward"],
        color="#4fc3f7",
        linestyle=":",
        linewidth=1.2,
        label="final deterministic eval",
    )
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(updates, ball_y, color="#4fc3f7", linewidth=1.2, label="ball y velocity")
    axes[1].axhline(-5.0, color="#f08a3c", linestyle="--", linewidth=1, label="target -5")
    axes[1].axhline(
        report["final_eval"]["mean_ball_y_velocity"],
        color="#8f8f8f",
        linestyle=":",
        linewidth=1,
        label="final deterministic eval",
    )
    axes[1].set_ylabel("ball y velocity")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.2)

    axes[2].plot(updates, policy_loss, color="#b388ff", linewidth=1.1, label="policy loss")
    axes[2].plot(updates, value_loss, color="#80cbc4", linewidth=1.1, label="value loss")
    axes[2].set_ylabel("loss")
    axes[2].set_xlabel("update")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.2)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
