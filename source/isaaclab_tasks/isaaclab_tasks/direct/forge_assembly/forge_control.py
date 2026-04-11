# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge Assembly: impedance control module.

Parameterized version of factory_control that supports variable DOF arms.
"""

import math

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    ee_pos,
    ee_quat,
    ee_linvel,
    ee_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_ee_pos,
    ctrl_target_ee_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    num_arm_joints=7,
    null_space_default_pos=None,
    dead_zone_thresholds=None,
):
    """Compute DOF torque for task-space impedance control.

    Supports any arm with configurable DOF count (6 or 7).
    """
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    pos_error, axis_angle_error = get_pose_error(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        ctrl_target_ee_pos=ctrl_target_ee_pos,
        ctrl_target_ee_quat=ctrl_target_ee_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_ee_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error
    task_wrench_motion = _apply_task_space_gains(
        delta_ee_pose=delta_ee_pose,
        ee_linvel=ee_linvel,
        ee_angvel=ee_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # Offset by dead zone
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Map task wrench to joint space: tau = J^T * wrench
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_torque = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)
    dof_torque[:, :num_arm_joints] = arm_torque

    # Null-space control (projects into null space of task-space)
    if null_space_default_pos is not None and num_arm_joints > 6:
        arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
        arm_mass_matrix_task = torch.inverse(
            jacobian @ arm_mass_matrix_inv @ jacobian_T
        )
        j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
        default_pos_tensor = torch.tensor(null_space_default_pos, device=device).repeat((num_envs, 1))
        distance = default_pos_tensor - dof_pos[:, :num_arm_joints]
        distance = (distance + math.pi) % (2 * math.pi) - math.pi  # normalize to [-pi, pi]
        u_null = cfg.ctrl.kd_null * -dof_vel[:, :num_arm_joints] + cfg.ctrl.kp_null * distance
        u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
        null_identity = torch.eye(num_arm_joints, device=device).unsqueeze(0)
        torque_null = (null_identity - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
        dof_torque[:, :num_arm_joints] += torque_null.squeeze(-1)

    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    ee_pos,
    ee_quat,
    ctrl_target_ee_pos,
    ctrl_target_ee_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target EE pose and current pose."""
    pos_error = ctrl_target_ee_pos - ee_pos

    if jacobian_type == "geometric":
        quat_dot = (ctrl_target_ee_quat * ee_quat).sum(dim=1, keepdim=True)
        ctrl_target_ee_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_ee_quat, -ctrl_target_ee_quat
        )
        ee_quat_norm = torch_utils.quat_mul(
            ee_quat, torch_utils.quat_conjugate(ee_quat)
        )[:, 0]
        ee_quat_inv = torch_utils.quat_conjugate(ee_quat) / ee_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_ee_quat, ee_quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported rotation error type: {rot_error_type}.")


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """Get delta DOF position from delta pose using specified IK method."""
    if ik_method == "pinv":
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":
        lambda_val = 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(min_singular_value < S, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_ee_pose, ee_linvel, ee_angvel, task_prop_gains, task_deriv_gains
):
    """Apply PD gains in task space."""
    task_wrench = torch.zeros_like(delta_ee_pose)
    lin_error = delta_ee_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - ee_linvel
    )
    rot_error = delta_ee_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - ee_angvel
    )
    return task_wrench
