# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the forge assembly (peg-in-hole) environment.

Custom termination term for peg-dropped detection. Standard time_out comes from
isaaclab.envs.mdp via wildcard import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_dropped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg,
    body_name: str = "panda_hand",
    distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate if peg is too far from end-effector (dropped).

    Optional safety termination. Direct forge does NOT use early termination
    (timeout only), but this can be enabled for unstable training scenarios.

    Args:
        env: The environment.
        asset_cfg: Scene entity config for the peg.
        robot_asset_cfg: Scene entity config for the robot.
        body_name: Name of the end-effector body.
        distance_threshold: Max allowed distance between peg and EE.

    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot: Articulation = env.scene[robot_asset_cfg.name]
    peg: RigidObject = env.scene[asset_cfg.name]

    body_ids, _ = robot.find_bodies(body_name)
    ee_body_idx = body_ids[0]

    ee_pos = robot.data.body_pos_w[:, ee_body_idx]
    peg_pos = peg.data.root_pos_w

    distance = torch.norm(peg_pos - ee_pos, p=2, dim=-1)
    return distance > distance_threshold
