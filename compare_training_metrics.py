#!/usr/bin/env python3
"""Analyze and compare UR10 vs Franka training metrics from tensorboard events."""

import os
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install tensorboard pandas matplotlib")
    sys.exit(1)


def load_tensorboard_events(log_dir: str, tags: list[str]) -> dict[str, pd.DataFrame]:
    """Load tensorboard events from a log directory."""
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}  # Load all events
    )
    ea.Reload()

    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame([{
                'step': e.step,
                'wall_time': e.wall_time,
                'value': e.value
            } for e in events])
        else:
            print(f"  Warning: Tag '{tag}' not found in {log_dir}")
            data[tag] = pd.DataFrame()

    return data


def plot_comparison(franka_data: dict, ur10_data: dict, tags: list[str], output_dir: str):
    """Plot comparison charts for Franka and UR10 training metrics."""
    os.makedirs(output_dir, exist_ok=True)

    tag_titles = {
        'losses/total_loss': 'Total Loss',
        'rewards/episode_reward': 'Episode Reward',
        'rewards/episode_success_rate': 'Success Rate',
        'rewards/episode_length': 'Episode Length',
        'metrics/episode_reward_mean': 'Mean Episode Reward',
    }

    for tag in tags:
        if tag not in tag_titles:
            tag_titles[tag] = tag

        fig, ax = plt.subplots(figsize=(12, 6))

        franka_df = franka_data.get(tag, pd.DataFrame())
        ur10_df = ur10_data.get(tag, pd.DataFrame())

        if not franka_df.empty:
            ax.plot(
                franka_df['step'],
                franka_df['value'],
                label='Franka',
                alpha=0.7,
                linewidth=1.5,
                color='blue'
            )

        if not ur10_df.empty:
            ax.plot(
                ur10_df['step'],
                ur10_df['value'],
                label='UR10',
                alpha=0.7,
                linewidth=1.5,
                color='orange'
            )

        ax.set_xlabel('Training Steps')
        ax.set_ylabel(tag_titles.get(tag, tag))
        ax.set_title(f'Training Comparison: Franka vs UR10 - {tag_titles.get(tag, tag)}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{tag.replace("/", "_")}_comparison.png'), dpi=150)
        plt.close()
        print(f"  Saved: {tag.replace('/', '_')}_comparison.png")


def print_summary_stats(franka_data: dict, ur10_data: dict, tags: list[str]):
    """Print summary statistics for comparison."""
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)

    for tag in tags:
        print(f"\n{tag}:")
        print("-" * 60)

        franka_df = franka_data.get(tag, pd.DataFrame())
        ur10_df = ur10_data.get(tag, pd.DataFrame())

        if not franka_df.empty:
            print(f"  Franka: min={franka_df['value'].min():.4f}, "
                  f"max={franka_df['value'].max():.4f}, "
                  f"mean={franka_df['value'].mean():.4f}, "
                  f"final={franka_df.iloc[-1]['value'] if len(franka_df) > 0 else 'N/A':.4f}")
        else:
            print("  Franka: No data")

        if not ur10_df.empty:
            print(f"  UR10:   min={ur10_df['value'].min():.4f}, "
                  f"max={ur10_df['value'].max():.4f}, "
                  f"mean={ur10_df['value'].mean():.4f}, "
                  f"final={ur10_df.iloc[-1]['value'] if len(ur10_df) > 0 else 'N/A':.4f}")
        else:
            print("  UR10:   No data (training may still be in progress)")


def main():
    # Default log directories
    franka_log_dir = "/home/an/IsaacLab/logs/rl_games/forge_assembly_franka/franka_peg_insert_exp1/summaries"
    ur10_log_dir = "/home/an/IsaacLab/logs/rl_games/forge_assembly_ur10/ur10_peg_insert_exp1/summaries"

    # Check for existing Franka data
    existing_franka_dirs = [
        "/home/an/IsaacLab/logs/rl_games/forge_assembly_franka/forge_assembly_franka/summaries",
        "/home/an/IsaacLab/logs/rl_games/Forge/test/summaries",
    ]

    print("Looking for training logs...")

    # Find available log directories
    franka_found = None
    ur10_found = None

    for d in existing_franka_dirs:
        if os.path.exists(d):
            franka_found = d
            print(f"  Found Franka logs: {d}")
            break

    if os.path.exists(ur10_log_dir):
        ur10_found = ur10_log_dir
        print(f"  Found UR10 logs: {ur10_log_dir}")
    else:
        # Search for UR10 logs
        for root, dirs, files in os.walk("/home/an/IsaacLab/logs/rl_games/"):
            if "ur10" in root.lower():
                if os.path.exists(os.path.join(root, "summaries")):
                    ur10_found = os.path.join(root, "summaries")
                    print(f"  Found UR10 logs: {ur10_found}")
                    break

    if not franka_found:
        print("No Franka training logs found. Please run training first.")
        franka_found = franka_log_dir

    if not ur10_found:
        print("No UR10 training logs found. Please run training first.")
        print("\nTo start UR10 training:")
        print("  bash train_ur10_forge.sh")
        return

    # Tags to analyze
    common_tags = [
        'losses/total_loss',
        'rewards/episode_reward',
        'metrics/episode_reward_mean',
    ]

    print(f"\nLoading Franka data from: {franka_found}")
    franka_data = load_tensorboard_events(franka_found, common_tags)

    print(f"Loading UR10 data from: {ur10_found}")
    ur10_data = load_tensorboard_events(ur10_found, common_tags)

    # Output directory for plots
    output_dir = "/home/an/IsaacLab/training_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating comparison plots in: {output_dir}")
    plot_comparison(franka_data, ur10_data, common_tags, output_dir)

    # Print summary statistics
    print_summary_stats(franka_data, ur10_data, common_tags)

    print("\n" + "=" * 80)
    print(f"Comparison complete! Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
