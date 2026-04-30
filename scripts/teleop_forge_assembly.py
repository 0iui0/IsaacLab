"""
Keyboard teleoperation for forge_assembly DirectRL environments.

Enables manual verification of peg-in-hole insertion feasibility:
  - Test whether tight-tolerance pairs (pair1/pair2) can be inserted
  - Observe contact forces during insertion
  - Verify fixed_asset stability under load
  - Debug collision geometry and friction parameters

Uses carb.input directly with keys that avoid Isaac Sim viewport conflicts.

Usage:
  # UR10 fixed-peg robot
  ./isaaclab.sh -p scripts/teleop_forge_assembly.py \\
      --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0

  # Franka gripper robot
  ./isaaclab.sh -p scripts/teleop_forge_assembly.py \\
      --task Isaac-ForgeAssembly-Franka-PegInsert-Direct-v0

  # CR5 fixed-peg robot
  ./isaaclab.sh -p scripts/teleop_forge_assembly.py \\
      --task Isaac-ForgeAssembly-CR5-PegInsert-Direct-v0

Controls (when teleop is ACTIVE, press SPACE or B to toggle):
  J/L : X+ / X- (left/right)
  I/K : Y- / Y+ (forward/backward)
  U/O : Z+ / Z- (up/down)
  N/M : yaw (rotate left/right)
  H   : toggle gripper (gripper robots only)
  R   : reset environment
  B/Space : toggle teleop ACTIVE/INACTIVE
"""

import argparse

from isaaclab.app import AppLauncher

# ---------- argparse ----------
parser = argparse.ArgumentParser(description="Keyboard teleop for forge_assembly")
parser.add_argument("--task", type=str, default="Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (1 for teleop)")
parser.add_argument("--asset_pair", type=int, default=None,
                    help="Asset pair index (0=factory, 1=pair1, 2=pair2). If unset, random each reset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app (MUST happen before any isaaclab imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- imports (after app launch) ----------
import weakref

import numpy as np

import carb
import omni
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401 — registers tasks
import isaaclab.utils.math as torch_utils
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_tasks.direct.forge_assembly import forge_utils  # noqa: E402


class ForgeKeyboard:
    """Keyboard controller for forge_assembly teleop.

    Uses carb.input with keys that avoid Isaac Sim viewport conflicts.

    Key bindings:
        J/L : X+ / X-
        I/K : Y- / Y+
        U/O : Z+ / Z-
        N/M : yaw (rotate left/right)
        H   : toggle gripper
    """

    _KEY_POS = {
        "J": np.array([1.0, 0.0, 0.0]),   # J → X+
        "L": np.array([-1.0, 0.0, 0.0]),  # L → X-
        "I": np.array([0.0, -1.0, 0.0]),  # I → Y-
        "K": np.array([0.0, 1.0, 0.0]),   # K → Y+
        "U": np.array([0.0, 0.0, 1.0]),   # U → Z+
        "O": np.array([0.0, 0.0, -1.0]),  # O → Z-
    }
    _KEY_ROT = {
        "N": np.array([0.0, 0.0, 1.0]),
        "M": np.array([0.0, 0.0, -1.0]),
    }
    _POS_KEYS = set(_KEY_POS.keys())
    _ROT_KEYS = set(_KEY_ROT.keys())

    def __init__(self, pos_sensitivity: float = 0.002, rot_sensitivity: float = 0.02, device: str = "cpu"):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._device = device

        # acquire omniverse interfaces
        self._input = carb.input.acquire_input_interface()
        # In some environments (Docker/VNC), the default app window may not be
        # available. Fall back to null keyboard if we can't get the window.
        self._keyboard = None
        self._keyboard_sub = None
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            if self._appwindow is not None:
                self._keyboard = self._appwindow.get_keyboard()
        except Exception:
            self._appwindow = None
        if self._keyboard is not None:
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
        else:
            print("[WARN] No keyboard device available. Teleop input disabled.")

        # command buffers
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._additional_callbacks = {}

        # pre-compute scaled deltas
        self._pos_delta = {k: v * self.pos_sensitivity for k, v in self._KEY_POS.items()}
        self._rot_delta = {k: v * self.rot_sensitivity for k, v in self._KEY_ROT.items()}

    def __del__(self):
        if self._keyboard_sub is not None and self._keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def reset(self):
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        command = np.concatenate([self._delta_pos, self._delta_rot])
        return torch.tensor(command, dtype=torch.float32, device=self._device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        # Carb API: event.input may be a string (newer API) or an object with .name
        raw = event.input
        key = raw if isinstance(raw, str) else raw.name
        etype = event.type

        # Debug: print non-movement key events
        if etype == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self._POS_KEYS and key not in self._ROT_KEYS:
                print(f"  [keyboard] event={key} type=PRESS")
        elif etype == carb.input.KeyboardEventType.KEY_RELEASE:
            if key not in self._POS_KEYS and key not in self._ROT_KEYS:
                print(f"  [keyboard] event={key} type=RELEASE")

        if etype == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._POS_KEYS:
                self._delta_pos += self._pos_delta[key]
            elif key in self._ROT_KEYS:
                self._delta_rot += self._rot_delta[key]
        elif etype == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self._POS_KEYS:
                self._delta_pos -= self._pos_delta[key]
            elif key in self._ROT_KEYS:
                self._delta_rot -= self._rot_delta[key]

        if etype == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._additional_callbacks:
                self._additional_callbacks[key]()

        return True


def main():
    # ---- create env ----
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device,
        num_envs=args_cli.num_envs, use_fabric=None,
    )
    # Extend episode length for teleoperation (default 10s is too short)
    env_cfg.episode_length_s = 600.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped  # ForgeEnv instance

    # Override EMA for responsive teleop (must be per-env tensor, not scalar)
    _ema_ones = torch.ones((base.num_envs, 1), device=base.device)
    base.ema_factor = _ema_ones.clone()

    # Enable collision visualisation in the viewport (green wireframe, useful for
    # verifying that static fixed assets use triangle mesh, not convex hull).
    carb.settings.get_settings().set_bool("/physics/showCollision", True)

    # ---- keyboard device ----
    keyboard = ForgeKeyboard(pos_sensitivity=0.002, rot_sensitivity=0.02, device=base.device)

    # ---- teleop state ----
    active = False
    step_count = 0
    gripper_closed = True

    # Original params saved/restored on toggle
    _saved_ctrl_params = {}

    # Gain multipliers (applied on top of env defaults); wrapped in a list for mutability in closures.
    _gain_scale = [1.0, 1.0]  # [kp_scale, kd_scale]

    def _apply_gain_multipliers():
        """Recompute prop/deriv gains from env defaults × current scales."""
        kps, kds = _gain_scale
        base.task_prop_gains[:] = base.default_gains * kps
        base.task_deriv_gains[:] = forge_utils.get_deriv_gains(base.default_gains) * kds

    def _apply_teleop_params():
        """Override env control params for responsive teleop.

        The env's default params (dead zone, position threshold, EMA) are
        designed for RL training stability and actively suppress small
        movements — exactly the opposite of what teleop needs.
        """
        base.dead_zone_thresholds[:] = 0.0                      # no dead zone
        base.pos_threshold[:] = pos_bounds                       # no per-step clip
        base.rot_threshold[:] = rot_threshold_teleop             # no per-step clip
        base.ema_factor = _ema_ones.clone()                      # no EMA smoothing
        _apply_gain_multipliers()

    def _on_gain_change():
        if active:
            _apply_gain_multipliers()
        print(f"  [Gains] Kp={_gain_scale[0]:.2f}x  Kd={_gain_scale[1]:.2f}x")

    # Maintained position/rotation actions — only updated when keys pressed
    held_pos_action = torch.zeros(3, device=base.device)
    held_rot_action = torch.zeros(3, device=base.device)

    # ---- helpers ----
    def _force_asset_pair():
        """Override asset pair if --asset_pair was specified."""
        if args_cli.asset_pair is not None:
            base.asset_pair_idx[:] = args_cli.asset_pair

    # ---- callbacks ----
    def on_reset():
        env.reset()
        _force_asset_pair()
        if active:
            # Reset re-randomizes params — re-apply teleop overrides
            _apply_teleop_params()
        held_pos_action.zero_()
        held_rot_action.zero_()
        # Re-capture from env after reset
        held_pos_action[:] = base.actions[0, :3]
        held_rot_action[:] = base.actions[0, 3:6]
        nonlocal step_count, gripper_closed
        step_count = 0
        gripper_closed = True
        print("  [Reset] env reset OK")

    def on_toggle():
        nonlocal active, held_pos_action, held_rot_action
        active = not active
        if active:
            # Save original params and override for teleop:
            #   - Zero null-space kp/kd so joint-centering doesn't interfere
            #   - Zero dead zone so every position error generates force
            #   - Relax thresholds to action bounds so per-step clip
            #     doesn't limit max force
            #   - No EMA smoothing for instant response
            _saved_ctrl_params["kp_null"] = base.cfg.ctrl.kp_null
            _saved_ctrl_params["kd_null"] = base.cfg.ctrl.kd_null
            base.cfg.ctrl.kp_null = 0.0
            base.cfg.ctrl.kd_null = 0.0
            _apply_teleop_params()
            # Capture current EE position as the held target for smooth startup
            fixed_obs = base.fixed_pos_obs_frame[0] + base.init_fixed_pos_obs_noise[0]
            current_rel = (base.ee_pos[0] - fixed_obs) / pos_bounds
            held_pos_action = current_rel.clamp(-1.0, 1.0)
            held_rot_action[:] = base.actions[0, 3:6]
        else:
            # Restore original control params
            if "kp_null" in _saved_ctrl_params:
                base.cfg.ctrl.kp_null = _saved_ctrl_params["kp_null"]
                base.cfg.ctrl.kd_null = _saved_ctrl_params["kd_null"]
        print(f"  [Teleop] {'ACTIVE' if active else 'INACTIVE'}")

    def on_gripper():
        nonlocal gripper_closed
        gripper_closed = not gripper_closed
        if base.gripper_slice is not None:
            base.ctrl_target_joint_pos[:, base.gripper_slice] = 0.0 if gripper_closed else 0.04
        print(f"  [Gripper] {'CLOSED' if gripper_closed else 'OPEN'}")

    keyboard.add_callback("R", on_reset)
    keyboard.add_callback("B", on_toggle)
    keyboard.add_callback("SPACE", on_toggle)
    keyboard.add_callback("H", on_gripper)
    keyboard.add_callback("Y", lambda: [_gain_scale.__setitem__(0, min(_gain_scale[0]*1.3, 5.0)), _on_gain_change()][-1])
    keyboard.add_callback("T", lambda: [_gain_scale.__setitem__(0, max(_gain_scale[0]/1.3, 0.1)), _on_gain_change()][-1])
    keyboard.add_callback("G", lambda: [_gain_scale.__setitem__(1, min(_gain_scale[1]*1.3, 5.0)), _on_gain_change()][-1])
    keyboard.add_callback("V", lambda: [_gain_scale.__setitem__(1, max(_gain_scale[1]/1.3, 0.1)), _on_gain_change()][-1])

    # ---- constants ----
    pos_bounds = torch.tensor(base.cfg.ctrl.pos_action_bounds, device=base.device)
    rot_threshold_teleop = torch.tensor([0.5, 0.5, 0.5], device=base.device)  # ~28° no clip
    dt = base.physics_dt * base.cfg.decimation

    has_sensor = base.force_sensor_body_idx is not None
    has_contact_sensor = base._contact_sensor.is_initialized
    fixed_peg = base.profile.grasp_type == "fixed_peg"
    has_gripper = base.profile.has_gripper

    # ---- print info ----
    print()
    print(f"{'='*60}")
    print(f"  Forge Assembly Teleop")
    print(f"  Task  : {args_cli.task}")
    print(f"  Robot : {type(base.profile).__name__}")
    print(f"  Grasp : {base.profile.grasp_type}")
    print(f"  Asset pairs: {base._num_asset_pairs}")
    if args_cli.asset_pair is not None:
        print(f"  Forced pair: {args_cli.asset_pair} (pair{args_cli.asset_pair})")
    else:
        print(f"  Pair selection: randomized")
    print(f"  Physics dt : {base.physics_dt*1000:.1f} ms")
    print(f"  Action dt  : {dt*1000:.1f} ms")
    print(f"{'='*60}")
    print()
    print("  Controls (when ACTIVE, press SPACE or B to toggle):")
    print("    B/Space → toggle ACTIVE/INACTIVE")
    print("    J/L → X+/-,  I/K → Y-/+,  U/O → Z+/-,  N/M → yaw")
    print("    H → gripper toggle,  R → reset")
    print("    Y/T → Kp +/-,  G/V → Kd +/-")
    print()

    # ---- initial reset ----
    env.reset()
    _force_asset_pair()
    base.ema_factor = _ema_ones.clone()
    held_pos_action[:] = base.actions[0, :3]     # capture initial position action
    held_rot_action[:] = base.actions[0, 3:6]    # capture initial rotation action

    # ---- main loop ----
    with torch.inference_mode():
        while simulation_app.is_running():
            cmd = keyboard.advance()

            # Build action
            action = torch.zeros((base.num_envs, 7), device=base.device)

            if active and cmd is not None:
                # Position: key delta in normalized space
                kp = cmd[:3] / pos_bounds  # (3,)
                if kp.norm() > 0:
                    # Key pressed: accumulate delta onto held position
                    held_pos_action = (held_pos_action + kp).clamp(-1.0, 1.0)
                else:
                    # No key: track current EE position to zero impedance-control force.
                    # Without this, the controller applies J^T*Kp*(ctrl_target - ee_pos)
                    # even when idle, causing unwanted drift.
                    fixed_obs = base.fixed_pos_obs_frame[0] + base.init_fixed_pos_obs_noise[0]
                    current_rel = (base.ee_pos[0] - fixed_obs) / pos_bounds
                    held_pos_action[:] = current_rel.clamp(-1.0, 1.0)
                action[:, :3] = held_pos_action.unsqueeze(0)

                # Rotation: yaw delta from N/M keys
                yaw_delta = cmd[5].item() * 0.15
                if abs(yaw_delta) > 0:
                    held_rot_action[2] = (held_rot_action[2] + yaw_delta).clamp(-1.0, 1.0)
                action[:, 3:6] = held_rot_action.unsqueeze(0)

                # Success prediction
                action[:, 6] = -1.0

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # ---- debug output ----
            step_count += 1
            if step_count % 15 == 0:
                ee = base.ee_pos[0].cpu()
                fixed = base.fixed_pos_obs_frame[0].cpu()

                xy_mm = float(torch.norm(ee[:2] - fixed[:2]).item() * 1000.0)
                z_mm = float((ee[2] - fixed[2]).item() * 1000.0)

                # Forces: try F/T sensor first, fall back to contact sensor
                if has_sensor:
                    f = base.force_sensor_smooth[0]
                    f_msg = (
                        f"F=[{f[0]:+6.1f} {f[1]:+6.1f} {f[2]:+6.1f}] "
                        f"T=[{f[3]:+5.1f} {f[4]:+5.1f} {f[5]:+5.1f}]"
                    )
                elif has_contact_sensor:
                    cf = base._contact_sensor.data.net_forces_w.sum(dim=1)[0]
                    f_msg = f"CF=[{cf[0]:+6.1f} {cf[1]:+6.1f} {cf[2]:+6.1f}]"
                else:
                    f_msg = "F=[N/A]"

                # Insertion depth (fixed_peg only)
                ins = ""
                if fixed_peg:
                    pt = base.peg_tip_pos[0].cpu()
                    ht = base.hole_top_pos[0].cpu()
                    ins = f" | insert={float(pt[2]-ht[2])*1000:+5.1f}mm"

                # Reward components
                rew_val = float(reward[0].item())

                status = "A" if active else "."
                print(
                    f"[{step_count:>5d} {status}] "
                    f"XY={xy_mm:6.2f}mm Z={z_mm:+6.2f}mm "
                    f"{f_msg} rew={rew_val:.3f}{ins}"
                )

                # ---- DEBUG: control signal diagnosis ----
                torque = base.joint_torque[0, :base.num_arm_joints].cpu()
                torque_norm = float(torque.norm().item())
                jac_norm = float(base.ee_jacobian[0].norm().item())
                # ctrl_target from impedance controller
                ctrl_pos = base.ee_pos[0].cpu()
                ctrl_target_pos = base.ee_pos[0].cpu()  # approximate; actual target from _apply_action
                pos_err = float((ctrl_pos - fixed).norm().item())
                action_val = f"[{base.actions[0,0]:+.3f} {base.actions[0,1]:+.3f} {base.actions[0,2]:+.3f}]"
                held = f"[{held_pos_action[0]:+.3f} {held_pos_action[1]:+.3f} {held_pos_action[2]:+.3f}]"

            # Reset if terminated
            if terminated.any() or truncated.any():
                env.reset()
                _force_asset_pair()
                if active:
                    # Reset re-randomizes params — re-apply teleop overrides
                    _apply_teleop_params()
                held_pos_action[:] = base.actions[0, :3]
                held_rot_action[:] = base.actions[0, 3:6]

    # ---- cleanup ----
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
