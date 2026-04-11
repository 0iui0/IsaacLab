#!/usr/bin/env python3
"""
Upload Isaac Lab RL-Games training metrics to Weights & Biases (wandb.ai).

This script extracts metrics from tensorboard event files and uploads them to wandb.

Usage:
    python upload_to_wandb.py --project FORGE_ASSEMBLY --run-name UR10_PegInsert_001 \
        --summary-dir /home/an/IsaacLab/logs/rl_games/Forge/test/summaries

Requirements:
    pip install wandb tensorboard
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Run: pip install wandb")
    exit(1)

from tensorboard.backend.event_processing import event_accumulator


def extract_tb_metrics(summary_dir: str) -> tuple[dict, list[dict]]:
    """Extract all metrics from tensorboard event files.

    Returns:
        Tuple of (config dict, list of metric steps)
    """
    files = [f for f in os.listdir(summary_dir) if f.startswith('events.out')]
    if not files:
        print(f"No tensorboard files found in {summary_dir}")
        return {}, []

    # Get the largest file (most data) instead of most recent (which might be empty)
    latest_file = max(files, key=lambda f: os.path.getsize(os.path.join(summary_dir, f)))
    print(f"Reading: {latest_file}")

    ea = event_accumulator.EventAccumulator(os.path.join(summary_dir, latest_file))
    ea.Reload()

    scalar_tags = ea.Tags().get('scalars', [])
    print(f"Found {len(scalar_tags)} scalar tags")

    # Extract all scalar metrics
    all_steps = []
    steps_by_tag = {}

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps_by_tag[tag] = len(events)
        for event in events:
            all_steps.append({
                'step': event.step,
                'wall_time': event.wall_time,
                tag: event.value
            })

    # Merge steps by step count
    merged_steps = {}
    for tag, events_count in steps_by_tag.items():
        events = ea.Scalars(tag)
        for event in events:
            step = event.step
            if step not in merged_steps:
                merged_steps[step] = {'step': step, 'wall_time': event.wall_time}
            merged_steps[step][tag] = event.value

    # Convert to sorted list
    sorted_steps = [merged_steps[step] for step in sorted(merged_steps.keys())]

    # Config metadata
    config = {
        'summary_dir': summary_dir,
        'total_steps': len(sorted_steps),
        'scalar_tags': len(scalar_tags),
        'tag_counts': steps_by_tag,
    }

    return config, sorted_steps


def upload_to_wandb(project: str, run_name: str, summary_dir: str,
                    entity: str = None, tags: list = None):
    """Upload training metrics to wandb."""

    # Initialize wandb
    wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        tags=tags or [],
        config={
            'summary_dir': summary_dir,
            'upload_time': datetime.now().isoformat(),
        }
    )

    # Extract metrics
    config, steps = extract_tb_metrics(summary_dir)

    # Add config to wandb
    wandb.config.update(config, allow_val_change=True)

    if not steps:
        print("No metrics to upload")
        wandb.finish()
        return

    # Log all metrics
    print(f"Uploading {len(steps)} steps...")
    for step_data in steps:
        step = step_data.pop('step')
        wall_time = step_data.pop('wall_time', None)

        # Separate metrics by type for better organization
        loss_metrics = {k: v for k, v in step_data.items() if 'loss' in k.lower()}
        reward_metrics = {k: v for k, v in step_data.items() if 'reward' in k.lower() or 'rew' in k.lower()}
        perf_metrics = {k: v for k, v in step_data.items() if 'performance' in k.lower() or 'fps' in k.lower() or 'time' in k.lower()}
        other_metrics = {k: v for k, v in step_data.items()
                        if k not in loss_metrics and k not in reward_metrics and k not in perf_metrics}

        # Log by category
        if loss_metrics:
            wandb.log({'loss/' + k.replace('losses/', ''): v for k, v in loss_metrics.items()}, step=step)

        if reward_metrics:
            wandb.log({'reward/' + k.replace('Episode/Episode_Reward/', '').replace('logs_rew_', ''): v
                      for k, v in reward_metrics.items()}, step=step)

        if perf_metrics:
            wandb.log({'performance/' + k.replace('performance/', ''): v for k, v in perf_metrics.items()}, step=step)

        if other_metrics:
            wandb.log({'metrics/' + k: v for k, v in other_metrics.items()}, step=step)

    print(f"Uploaded {len(steps)} steps to wandb")
    print(f"Run URL: {wandb.run.get_url()}")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Upload RL-Games metrics to wandb')
    parser.add_argument('--project', type=str, default='forge-assembly',
                       help='Wandb project name')
    parser.add_argument('--run-name', type=str, required=True,
                       help='Wandb run name')
    parser.add_argument('--summary-dir', type=str, required=True,
                       help='Path to tensorboard summaries directory')
    parser.add_argument('--entity', type=str, default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--tags', type=str, nargs='*', default=None,
                       help='Tags for the run')
    parser.add_argument('--robot', type=str, default='unknown',
                       help='Robot type (UR10, Franka, CR5)')

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.summary_dir):
        print(f"Error: Summary directory not found: {args.summary_dir}")
        return

    # Set default tags
    tags = args.tags or []
    tags.extend([args.robot, 'forge-assembly', 'rl-games'])

    print(f"Uploading to wandb...")
    print(f"  Project: {args.project}")
    print(f"  Run: {args.run_name}")
    print(f"  Robot: {args.robot}")
    print(f"  Summary dir: {args.summary_dir}")
    print(f"  Tags: {tags}")

    upload_to_wandb(
        project=args.project,
        run_name=args.run_name,
        summary_dir=args.summary_dir,
        entity=args.entity,
        tags=tags
    )


if __name__ == '__main__':
    main()
