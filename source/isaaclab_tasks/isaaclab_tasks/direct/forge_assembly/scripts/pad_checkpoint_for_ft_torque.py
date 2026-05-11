"""Pad a 24D-obs checkpoint to 27D by appending 3 zero columns for ft_torque.

Usage:
    python pad_checkpoint_for_ft_torque.py \
        --input <24D_checkpoint.pth> \
        --output <27D_checkpoint.pth>

The padded checkpoint preserves all learned weights. The 3 new torque
input dimensions start with zero weights, so the network behaves
identically at step 0 and gradually learns to use torque information.
"""
import argparse
import copy

import torch


def pad_checkpoint(in_path: str, out_path: str, extra_dims: int = 3) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    model = ckpt["model"]

    # 1) Input running mean/std: (24,) -> (24+3,)
    key_rms = "running_mean_std"
    for suffix in ("running_mean", "running_var"):
        k = f"{key_rms}.{suffix}"
        old = model[k]
        if old.shape[0] != 24:
            raise ValueError(f"Expected {k} dim 0 = 24, got {old.shape[0]}")
        pad = torch.zeros(extra_dims)
        model[k] = torch.cat([old, pad], dim=0)

    # count stays scalar, no change needed

    # 2) LSTM layer-0 input weight: (4096, 24) -> (4096, 27)
    key_w = "a2c_network.rnn.rnn.weight_ih_l0"
    old_w = model[key_w]
    if old_w.shape[1] != 24:
        raise ValueError(f"Expected {key_w} dim 1 = 24, got {old_w.shape[1]}")
    pad_w = torch.zeros(old_w.shape[0], extra_dims)
    model[key_w] = torch.cat([old_w, pad_w], dim=1)

    # All other layers: input dim is hidden_size (1024), unchanged.
    # Print summary
    print("Padded layers:")
    print(f"  running_mean_std.running_mean: (24,) -> ({model[f'{key_rms}.running_mean'].shape[0]},)")
    print(f"  running_mean_std.running_var:  (24,) -> ({model[f'{key_rms}.running_var'].shape[0]},)")
    print(f"  weight_ih_l0: ({old_w.shape[0]}, 24) -> ({model[key_w].shape[0]}, {model[key_w].shape[1]})")

    torch.save(ckpt, out_path)
    print(f"\nSaved padded checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-dims", type=int, default=3)
    args = parser.parse_args()
    pad_checkpoint(args.input, args.output, args.extra_dims)
