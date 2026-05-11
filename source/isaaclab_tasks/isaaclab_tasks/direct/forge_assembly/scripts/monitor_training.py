"""Monitor Forge Ablation B/C training progress from TensorBoard logs."""

import argparse
import os
import subprocess

EXPERIMENTS = {
    "B": {
        "log_dir": "/workspace/isaaclab/logs/rl_games/Forge_Ablation_B/ablation_B_3axis_force",
        "container": "isaac-lab-task-main",
        "gpu": 0,
        "max_epochs": 100,
        "pattern": "AblationB",
    },
    "C": {
        "log_dir": "/workspace/isaaclab/logs/rl_games/Forge_Ablation_C/ablation_C_6axis_ft",
        "container": "isaac-lab-task-c",
        "gpu": 1,
        "max_epochs": 100,
        "pattern": "AblationC",
    },
}


def _read_latest(tag, ea, available):
    if tag not in available:
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    e = events[-1]
    return {"step": e.step, "value": e.value}


def check_process(container, pattern):
    r = subprocess.run(
        ["docker", "exec", container, "pgrep", "-f", f"train.py.*{pattern}"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() != ""


def monitor_experiment(name, cfg):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    summaries_dir = os.path.join(cfg["log_dir"], "summaries")
    running = check_process(cfg["container"], cfg["pattern"])
    status = "RUNNING" if running else "STOPPED"

    print(f"  [{name}] Status: {status}")

    if not os.path.exists(summaries_dir):
        print(f"  [{name}] No data yet.")
        return

    ea = EventAccumulator(summaries_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    available = set(tags)
    if not tags:
        print(f"  [{name}] No TensorBoard data yet.")
        return

    metrics = {}

    # Core metrics
    for tag, key in [
        ("rewards/iter", "reward"),
        ("successes/iter", "success_rate"),
        ("info/epochs", "epoch"),
        ("performance/step_fps", "fps"),
        ("shaped_rewards/iter", "shaped_reward"),
        ("episode_lengths/iter", "ep_length"),
    ]:
        v = _read_latest(tag, ea, available)
        if v:
            metrics[key] = v["value"]

    # Reward components
    for comp in ["kp_baseline", "kp_coarse", "kp_fine", "action_penalty_ee",
                 "action_grad_penalty", "curr_engaged", "curr_success",
                 "action_penalty_asset", "contact_penalty"]:
        key = f"logs_rew_{comp}/iter"
        v = _read_latest(key, ea, available)
        if v:
            metrics[f"rew/{comp}"] = v["value"]

    # F/T sensor data
    for ft in ["fx", "fy", "fz", "tx", "ty", "tz", "force_norm", "torque_norm"]:
        key = f"ft_sensor/{ft}/iter"
        v = _read_latest(key, ea, available)
        if v:
            metrics[f"ft/{ft}"] = v["value"]

    # Print
    epoch = metrics.get("epoch", "?")
    print(f"  [{name}] Epoch: {epoch}/{cfg['max_epochs']}  Reward: {metrics.get('reward', '?'):.2f}  "
          f"Success: {metrics.get('success_rate', 0):.2%}  FPS: {metrics.get('fps', '?'):.1f}")

    # F/T sensor
    ft_keys = [k for k in metrics if k.startswith("ft/")]
    if ft_keys:
        ft_str = "  ".join(f"{k.split('/')[1]}={metrics[k]:.4f}" for k in ft_keys)
        print(f"  [{name}] FT: {ft_str}")
    else:
        print(f"  [{name}] FT: no ft_sensor data yet")

    # Key reward components
    for k in ["rew/kp_fine", "rew/contact_penalty", "rew/curr_success"]:
        if k in metrics:
            pass  # already collected

    # Checkpoints
    nn_dir = os.path.join(cfg["log_dir"], "nn")
    if os.path.isdir(nn_dir):
        ckpts = [f for f in os.listdir(nn_dir) if f.endswith(".pth")]
        if ckpts:
            print(f"  [{name}] Checkpoints: {len(ckpts)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["B", "C", "all"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("Forge Ablation Training Monitor")
    print("=" * 60)

    # GPU status
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"\nGPU: {r.stdout.strip()}")

    experiments = EXPERIMENTS if args.exp == "all" else {args.exp: EXPERIMENTS[args.exp]}
    print()
    for name, cfg in experiments.items():
        monitor_experiment(name, cfg)
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
