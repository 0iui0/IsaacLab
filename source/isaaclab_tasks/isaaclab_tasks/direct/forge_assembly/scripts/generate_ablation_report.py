"""Generate ablation experiment report from TensorBoard logs.

Usage:
    python generate_ablation_report.py --output report.md

Reads TensorBoard events from A, B, C experiments and generates
a comparison report with reward curves, success rates, and F/T analysis.
"""

import argparse
import os

EXPERIMENTS = {
    "A (baseline, ft=0)": "/workspace/isaaclab/logs/rl_games/Forge_Pair123/finetune_pair123/summaries",
    "B (3-axis force)": "/workspace/isaaclab/logs/rl_games/Forge_Ablation_B/ablation_B_3axis_force/summaries",
    "C (6-axis F/T)": "/workspace/isaaclab/logs/rl_games/Forge_Ablation_C/ablation_C_6axis_ft/summaries",
}


def read_metric_series(ea, tag):
    if tag not in set(ea.Tags().get("scalars", [])):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ablation_report.md")
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    lines = []
    lines.append("# F/T Ablation Experiment Report\n")
    lines.append("| Experiment | Obs Dim | Force Input | Epochs | Best Reward | Best Success | Final Reward | Final Success |")
    lines.append("|---|---|---|---|---|---|---|---|")

    all_data = {}

    for name, path in EXPERIMENTS.items():
        if not os.path.exists(path):
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
            continue

        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()

        rewards = read_metric_series(ea, "rewards/iter")
        successes = read_metric_series(ea, "successes/iter")
        epochs = read_metric_series(ea, "info/epochs")

        if not rewards:
            lines.append(f"| {name} | - | - | 0 | - | - | - | - |")
            continue

        best_reward = max(r[1] for r in rewards)
        best_success = max(s[1] for s in successes) if successes else 0
        final_reward = rewards[-1][1]
        final_success = successes[-1][1] if successes else 0
        total_epochs = int(epochs[-1][1]) if epochs else 0

        lines.append(f"| {name} | - | - | {total_epochs} | {best_reward:.2f} | {best_success:.2%} | {final_reward:.2f} | {final_success:.2%} |")

        all_data[name] = {
            "rewards": rewards,
            "successes": successes,
            "ft": {},
        }

        # Read F/T data
        for ft in ["fx", "fy", "fz", "tx", "ty", "tz", "force_norm", "torque_norm"]:
            series = read_metric_series(ea, f"ft_sensor/{ft}/iter")
            if series:
                all_data[name]["ft"][ft] = series

    lines.append("")

    # F/T Analysis section
    lines.append("## Force/Torque Sensor Analysis\n")
    for name, data in all_data.items():
        if data["ft"]:
            lines.append(f"### {name}\n")
            lines.append("| Metric | Mean | Max | Min |")
            lines.append("|---|---|---|---|")
            for ft, series in data["ft"].items():
                vals = [s[1] for s in series]
                lines.append(f"| {ft} | {sum(vals)/len(vals):.4f} | {max(vals):.4f} | {min(vals):.4f} |")
            lines.append("")

    # Reward curve data (for external plotting)
    lines.append("## Reward Curve Data (CSV)\n")
    lines.append("```")
    lines.append("epoch," + ",".join(all_data.keys()))
    if all_data:
        max_len = max(len(d["rewards"]) for d in all_data.values())
        for i in range(max_len):
            row = [str(i + 1)]
            for name, data in all_data.items():
                if i < len(data["rewards"]):
                    row.append(f"{data['rewards'][i][1]:.2f}")
                else:
                    row.append("")
            lines.append(",".join(row))
    lines.append("```\n")

    # Conclusion template
    lines.append("## Conclusions\n")
    lines.append("- [ ] Fill in: Which force modality improves success rate the most?")
    lines.append("- [ ] Fill in: Does torque information help with alignment?")
    lines.append("- [ ] Fill in: Diminishing returns analysis (B vs C)")
    lines.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
