#!/usr/bin/env python3
"""
Compare UR10 vs Franka training metrics for Forge Assembly peg insertion task.

This script extracts and compares training metrics from tensorboard summaries
for both robot configurations.

Usage:
    python compare_ur10_franka.py [--franka-dir FRANKA_DIR] [--ur10-dir UR10_DIR]
"""

import argparse
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics(summary_dir: str) -> dict:
    """Extract training metrics from tensorboard summaries."""
    import os

    files = [f for f in os.listdir(summary_dir) if f.startswith('events.out')]
    if not files:
        return {}

    # Get the largest file (most complete training run)
    largest_file = max(files, key=lambda f: os.path.getsize(os.path.join(summary_dir, f)))
    ea = event_accumulator.EventAccumulator(os.path.join(summary_dir, largest_file))
    ea.Reload()

    scalar_tags = ea.Tags().get('scalars', [])
    metrics = {'file': largest_file, 'tags': len(scalar_tags)}

    # Extract key metrics
    reward_tags = [t for t in scalar_tags if 'Episode_Reward' in t]
    for tag in reward_tags:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            metrics[f'reward/{tag}'] = {
                'steps': len(events),
                'final': vals[-1],
                'max': max(vals),
                'mean': sum(vals) / len(vals)
            }

    loss_tags = [t for t in scalar_tags if t.startswith('losses/')]
    for tag in loss_tags:
        events = ea.Scalars(tag)
        if events:
            vals = [e.value for e in events]
            metrics[f'loss/{tag}'] = {
                'steps': len(events),
                'final': vals[-1],
                'mean': sum(vals) / len(vals)
            }

    perf_tags = [t for t in scalar_tags if t.startswith('performance/')]
    for tag in perf_tags:
        events = ea.Scalars(tag)
        if events:
            metrics[f'perf/{tag}'] = events[-1].value

    return metrics


def print_comparison(franka_metrics: dict, ur10_metrics: dict):
    """Print side-by-side comparison of metrics."""
    print("\n" + "=" * 80)
    print("FRANKA vs UR10 TRAINING METRICS COMPARISON")
    print("=" * 80)

    print(f"\nFranka file: {franka_metrics.get('file', 'N/A')}")
    print(f"UR10 file: {ur10_metrics.get('file', 'N/A')}")

    # Compare reward metrics
    print("\n--- Episode Rewards (final values) ---")
    reward_keys = set()
    for k in franka_metrics:
        if k.startswith('reward/'):
            reward_keys.add(k.replace('reward/', ''))

    for key in sorted(reward_keys):
        franka_key = f'reward/{key}'
        ur10_key = f'reward/{key}'

        franka_val = franka_metrics.get(franka_key, {})
        ur10_val = ur10_metrics.get(ur10_key, {})

        franka_final = franka_val.get('final', 'N/A')
        ur10_final = ur10_val.get('final', 'N/A')

        print(f"{key}:")
        print(f"  Franka: {franka_final:.6f}" if isinstance(franka_final, float) else f"  Franka: {franka_final}")
        print(f"  UR10:   {ur10_final:.6f}" if isinstance(ur10_final, float) else f"  UR10:   {ur10_final}")

    # Compare loss metrics
    print("\n--- Losses (final values) ---")
    loss_keys = set()
    for k in franka_metrics:
        if k.startswith('loss/'):
            loss_keys.add(k.replace('loss/', ''))

    for key in sorted(loss_keys):
        franka_key = f'loss/{key}'
        ur10_key = f'loss/{key}'

        franka_val = franka_metrics.get(franka_key, {})
        ur10_val = ur10_metrics.get(ur10_key, {})

        franka_final = franka_val.get('final', 'N/A')
        ur10_final = ur10_val.get('final', 'N/A')

        print(f"{key}:")
        print(f"  Franka: {franka_final:.6f}" if isinstance(franka_final, float) else f"  Franka: {franka_final}")
        print(f"  UR10:   {ur10_final:.6f}" if isinstance(ur10_final, float) else f"  UR10:   {ur10_final}")

    # Compare performance
    print("\n--- Performance ---")
    perf_keys = set()
    for k in franka_metrics:
        if k.startswith('perf/'):
            perf_keys.add(k.replace('perf/', ''))

    for key in sorted(perf_keys):
        franka_key = f'perf/{key}'
        ur10_key = f'perf/{key}'

        franka_val = franka_metrics.get(franka_key)
        ur10_val = ur10_metrics.get(ur10_key)

        print(f"{key}:")
        print(f"  Franka: {franka_val:.1f}" if isinstance(franka_val, float) else f"  Franka: {franka_val}")
        print(f"  UR10:   {ur10_val:.1f}" if isinstance(ur10_val, float) else f"  UR10:   {ur10_val}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare UR10 vs Franka training metrics')
    parser.add_argument('--franka-dir', type=str,
                        default='/home/an/IsaacLab/logs/rl_games/forge_assembly_franka/forge_assembly_franka/summaries',
                        help='Franka training summaries directory')
    parser.add_argument('--ur10-dir', type=str,
                        default='/home/an/IsaacLab/logs/rl_games/Forge/test/summaries',
                        help='UR10 training summaries directory')
    args = parser.parse_args()

    print("Extracting Franka metrics...")
    franka_metrics = extract_metrics(args.franka_dir)
    if franka_metrics:
        print(f"  Found {franka_metrics['tags']} scalar tags")
    else:
        print("  No metrics found!")

    print("Extracting UR10 metrics...")
    ur10_metrics = extract_metrics(args.ur10_dir)
    if ur10_metrics:
        print(f"  Found {ur10_metrics['tags']} scalar tags")
    else:
        print("  No metrics found (UR10 training may not have produced data yet)")

    if franka_metrics and ur10_metrics:
        print_comparison(franka_metrics, ur10_metrics)
    elif franka_metrics:
        print("\nOnly Franka metrics available. UR10 training data not found.")


if __name__ == '__main__':
    main()
