"""Monitor Forge Pair12 training progress from TensorBoard logs."""

import os
import subprocess

LOG_DIR = "/workspace/isaaclab/logs/rl_games/Forge_Pair123/finetune_pair123"
MAX_EPOCHS = 500


def _read_latest(tag: str, ea, available_tags: set) -> dict | None:
    if tag not in available_tags:
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    e = events[-1]
    return {"step": e.step, "value": e.value}


def check_process():
    r = subprocess.run(
        ["pgrep", "-f", "train.py.*Pair12"], capture_output=True, text=True
    )
    return r.stdout.strip() != ""


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    print("=" * 60)
    print("Forge Pair12 Training Monitor")
    print("=" * 60)

    running = check_process()
    status = "RUNNING" if running else "STOPPED"
    print(f"Process: {status}")

    # Point to summaries dir where event files actually live
    summaries_dir = os.path.join(LOG_DIR, "summaries")
    ea = EventAccumulator(summaries_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    available = set(tags)
    if not tags:
        print("No data yet (training may still be initializing).")
        print("=" * 60)
        return

    # -- Key metrics --
    metrics = {}

    # Reward
    r = _read_latest("rewards/iter", ea, available)
    if r:
        metrics["reward/mean"] = r["value"]
        metrics["reward/step"] = r["step"]

    sr = _read_latest("shaped_rewards/iter", ea, available)
    if sr:
        metrics["shaped_reward/mean"] = sr["value"]

    # Success rate
    s = _read_latest("successes/iter", ea, available)
    if s:
        metrics["success_rate"] = s["value"]

    st = _read_latest("success_times/iter", ea, available)
    if st:
        metrics["success_time/mean"] = st["value"]

    # Reward components
    for comp in ["kp_baseline", "kp_coarse", "kp_fine", "action_penalty_ee",
                 "action_grad_penalty", "curr_engaged", "curr_success",
                 "action_penalty_asset", "contact_penalty", "success_pred_error"]:
        key = f"logs_rew_{comp}/iter"
        v = _read_latest(key, ea, available)
        if v:
            metrics[f"rew/{comp}"] = v["value"]

    # Losses
    for loss in ["a_loss", "c_loss", "entropy", "bounds_loss", "cval_loss"]:
        v = _read_latest(f"losses/{loss}", ea, available)
        if v:
            metrics[f"loss/{loss}"] = v["value"]

    # Learning rate
    lr = _read_latest("info/last_lr", ea, available)
    if lr:
        metrics["lr"] = lr["value"]

    kl = _read_latest("info/kl", ea, available)
    if kl:
        metrics["kl"] = kl["value"]

    # Epoch
    ep = _read_latest("info/epochs", ea, available)
    if ep:
        metrics["epoch"] = ep["value"]

    # Performance
    fps = _read_latest("performance/step_fps", ea, available)
    if fps:
        metrics["step_fps"] = fps["value"]

    # Episode length
    el = _read_latest("episode_lengths/iter", ea, available)
    if el:
        metrics["episode_length/mean"] = el["value"]

    # Early termination precision/recall at 0.5
    for thresh in ["0.5", "0.7", "0.9"]:
        p = _read_latest(f"early_term_precision/{thresh}/iter", ea, available)
        rc = _read_latest(f"early_term_recall/{thresh}/iter", ea, available)
        if p:
            metrics[f"early_prec/{thresh}"] = p["value"]
        if rc:
            metrics[f"early_recall/{thresh}"] = rc["value"]

    # -- Print --
    epoch = metrics.get("epoch", "?")
    print(f"\nEpoch: {epoch} / {MAX_EPOCHS}")
    print(f"Step FPS: {metrics.get('step_fps', '?')}")

    print(f"\n{'--- Rewards ---':^60}")
    for k in ["reward/mean", "shaped_reward/mean", "success_rate",
              "success_time/mean", "episode_length/mean"]:
        if k in metrics:
            v = metrics[k]
            print(f"  {k:<35} {v:>12.4f}")

    print(f"\n{'--- Reward Components ---':^60}")
    for k, v in sorted(metrics.items()):
        if k.startswith("rew/"):
            print(f"  {k:<35} {v:>12.6f}")

    print(f"\n{'--- Losses & Training ---':^60}")
    for k in ["loss/a_loss", "loss/c_loss", "loss/cval_loss", "loss/entropy",
              "loss/bounds_loss", "lr", "kl"]:
        if k in metrics:
            v = metrics[k]
            print(f"  {k:<35} {v:>12.6f}")

    print(f"\n{'--- Early Termination ---':^60}")
    for k, v in sorted(metrics.items()):
        if k.startswith("early_"):
            print(f"  {k:<35} {v:>12.4f}")

    # Checkpoints
    nn_dir = os.path.join(LOG_DIR, "nn")
    if os.path.isdir(nn_dir):
        ckpts = [f for f in os.listdir(nn_dir) if f.endswith(".pth")]
        if ckpts:
            print(f"\nCheckpoints saved: {len(ckpts)} files")
            for f in sorted(ckpts):
                size = os.path.getsize(os.path.join(nn_dir, f))
                print(f"  {f} ({size / 1024 / 1024:.1f} MB)")

    if not running:
        print("\n*** Training appears to have STOPPED. ***")

    print("=" * 60)


if __name__ == "__main__":
    main()
