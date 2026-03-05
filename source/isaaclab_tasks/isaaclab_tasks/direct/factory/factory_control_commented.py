# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: control module.

This module implements task-space impedance control for the Franka Emika Panda robot.
It provides:
1. Task-space PD control (operational space control)
2. Jacobian transpose mapping (from Cartesian to joint space)
3. Null-space control (secondary objectives)
4. Dead-zone compensation (sensor unreliability simulation)

References:
- ETH Zurich Robot Dynamics Lecture Notes: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
- Modern Robotics: http://hades.mech.northwestern.edu/figures/figures.php
- RSS 2007 Paper: https://roboticsproceedings.org/rss07/p31.pdf
"""

import math

import torch

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.math import axis_angle_from_quat


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
):
    """
    Compute Franka DOF torque to move fingertips towards target pose using impedance control.

    This function implements task-space impedance control with the following steps:
    1. Compute pose error (position and orientation)
    2. Apply PD control law in task space
    3. Apply dead-zone compensation
    4. Map to joint space via Jacobian transpose
    5. Add null-space control for secondary objectives

    Args:
        cfg: Environment configuration
        dof_pos: Current joint positions [num_envs, num_dof]
        dof_vel: Current joint velocities [num_envs, num_dof]
        fingertip_midpoint_pos: Current fingertip position [num_envs, 3]
        fingertip_midpoint_quat: Current fingertip orientation [num_envs, 4]
        fingertip_midpoint_linvel: Current linear velocity [num_envs, 3]
        fingertip_midpoint_angvel: Current angular velocity [num_envs, 3]
        jacobian: Geometric Jacobian matrix [num_envs, 6, 7]
        arm_mass_matrix: Arm mass matrix [num_envs, 7, 7]
        ctrl_target_fingertip_midpoint_pos: Target position [num_envs, 3]
        ctrl_target_fingertip_midpoint_quat: Target orientation [num_envs, 4]
        task_prop_gains: Proportional gains Kp [num_envs, 6]
        task_deriv_gains: Derivative gains Kd [num_envs, 6]
        device: Device to create tensors on
        dead_zone_thresholds: Force/torque dead zone thresholds [num_envs, 6]

    Returns:
        dof_torque: Joint torques [num_envs, num_dof]
        task_wrench: Task-space wrench (force/torque) [num_envs, 6]
    """
    # Reference: ETH Zurich Robot Dynamics Lecture Notes, Equations 3.96-3.98
    # https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf

    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    # ============================================================================
    # Step 1: Compute Pose Error
    # ============================================================================
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    # Concatenate position and rotation errors into 6D twist
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # ============================================================================
    # Step 2: Apply Task-Space PD Control Law
    # ============================================================================
    # Equation: tau = Kp * error - Kd * velocity
    # This implements a virtual spring-damper system in task space
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # ============================================================================
    # Step 3: Apply Dead-Zone Compensation
    # ============================================================================
    # This simulates sensor unreliability at low forces
    # If force is below threshold, output 0
    # If force is above threshold, subtract threshold (hysteresis)
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # ============================================================================
    # Step 4: Map to Joint Space via Jacobian Transpose
    # ============================================================================
    # Equation: tau_joint = J^T * F_task
    # This maps Cartesian forces/torques to joint torques using virtual work principle
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # ============================================================================
    # Step 5: Add Null-Space Control
    # ============================================================================
    # Null-space control optimizes secondary objectives without affecting task space
    # Examples: maintain default posture, minimize energy, avoid singularities
    #
    # Reference: https://roboticsproceedings.org/rss07/p31.pdf

    # Compute inverse mass matrix
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)

    # Compute task-space mass matrix (ETH eq. 3.86)
    # Lambda_task = (J * M^{-1} * J^T)^{-1}
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )

    # Compute dynamically consistent inverse Jacobian
    # J_eef_inv = Lambda_task * J * M^{-1}
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    # Default joint positions (for secondary task)
    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))

    # Null-space control: move towards default posture
    # Equation: u_null = Kp_null * (q_default - q) - Kd_null * q_dot
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]
    u_null = cfg.ctrl.kp_null * distance_to_default_dof_pos - cfg.ctrl.kd_null * dof_vel[:, :7]

    # Project into null space: (I - J^T * J_inv) * u_null
    # This ensures null-space control doesn't affect task space motion
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null

    # Add null-space torque to primary task-space torque
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # ============================================================================
    # Step 6: Clamp Torques
    # ============================================================================
    # Limit joint torques to safe range for Franka Emika Panda
    # TODO: Verify it's okay to no longer do gripper control here.
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """
    Compute task-space error between target and current fingertip pose.

    This function computes the error in SE(3) (Special Euclidean Group) considering:
    - Position error: straightforward Euclidean distance
    - Rotation error: quaternion or axis-angle representation

    Args:
        fingertip_midpoint_pos: Current position [num_envs, 3]
        fingertip_midpoint_quat: Current orientation (quaternion) [num_envs, 4]
        ctrl_target_fingertip_midpoint_pos: Target position [num_envs, 3]
        ctrl_target_fingertip_midpoint_quat: Target orientation (quaternion) [num_envs, 4]
        jacobian_type: Type of Jacobian ("geometric" or "analytic")
        rot_error_type: Rotation error representation ("quat" or "axis_angle")

    Returns:
        pos_error: Position error [num_envs, 3]
        rot_error: Rotation error (quaternion or axis-angle) [num_envs, 3 or 4]

    Reference:
        https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
        (See example 2.9.8 for geometric Jacobian)
    """
    # ============================================================================
    # Position Error
    # ============================================================================
    # Simple Euclidean distance in Cartesian space
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # ============================================================================
    # Rotation Error
    # ============================================================================
    if jacobian_type == "geometric":
        # Geometric Jacobian uses quaternion algebra
        # Reference: https://personal.utdallas.edu/~sxb027100/dock/quat.html

        # Check for shortest path rotation using quaternion dot product
        # This ensures we take the shortest rotation path (avoiding >180° rotations)
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0,
            ctrl_target_fingertip_midpoint_quat,  # Use this quaternion
            -ctrl_target_fingertip_midpoint_quat,  # Or its negative (shortest path)
        )

        # Compute quaternion inverse: q^(-1) = q* / ||q||²
        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[:, 0]  # scalar component = w (real part)
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)

        # Compute quaternion error: q_error = q_target * q_current^(-1)
        # This represents the rotation from current to target orientation
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle representation (axis * theta)
        # This is more numerically stable for small rotations
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported rotation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    """
    Get delta Franka DOF position from delta pose using specified IK method.

    This function computes the joint space displacement required to achieve a given
    task-space displacement, using various inverse kinematics methods.

    Args:
        delta_pose: Desired Cartesian space displacement [num_envs, 6]
        ik_method: Inverse kinematics method ("pinv", "trans", "dls", "svd")
        jacobian: Geometric Jacobian matrix [num_envs, 6, 7]
        device: Device to create tensors on

    Returns:
        delta_dof_pos: Joint space displacement [num_envs, 7]

    References:
        1. IK Survey: https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        2. ETH Zurich (p. 47): https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    """

    if ik_method == "pinv":
        # ==========================================================================
        # Jacobian Pseudoinverse (Moore-Penrose)
        # ==========================================================================
        # Equation: q̇ = J^+ * v
        # where J^+ = V * S^(-1) * U^T (from SVD)
        #
        # Pros: Minimum norm solution
        # Cons: Can have large joint velocities near singularities
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":
        # ==========================================================================
        # Jacobian Transpose Method
        # ==========================================================================
        # Equation: q̇ = J^T * v
        #
        # Pros: Simple, computationally efficient
        # Cons: Only works for small displacements, not exact
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":
        # ==========================================================================
        # Damped Least Squares (Levenberg-Marquardt)
        # ==========================================================================
        # Equation: q̇ = J^T * (J * J^T + λ²I)^(-1) * v
        #
        # Pros: Handles singularities gracefully
        # Cons: Requires tuning lambda parameter
        lambda_val = 0.1  # Damping factor
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":
        # ==========================================================================
        # Adaptive SVD (Singular Value Decomposition)
        # ==========================================================================
        # Equation: q̇ = V * S^(-1) * U^T * v
        # where J = U * S * V^T (from SVD)
        #
        # Pros: Handles singularities, adaptive threshold
        # Cons: More complex computation
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        # Only invert singular values above threshold
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def _apply_task_space_gains(
    delta_fingertip_pose,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    task_prop_gains,
    task_deriv_gains,
):
    """
    Apply task-space PD gains to pose error.

    This implements the control law:
    F = Kp * error - Kd * velocity
    τ = Kp_rot * error_rot - Kd_rot * angular_velocity

    which is a virtual spring-damper system in task space.

    Args:
        delta_fingertip_pose: Pose error [num_envs, 6]
                          - [:3] = position error (meters)
                          - [3:6] = rotation error (axis-angle, radians)
        fingertip_midpoint_linvel: Current linear velocity [num_envs, 3]
        fingertip_midpoint_angvel: Current angular velocity [num_envs, 3]
        task_prop_gains: Proportional gains Kp [num_envs, 6]
        task_deriv_gains: Derivative gains Kd [num_envs, 6]

    Returns:
        task_wrench: Desired force/torque in task space [num_envs, 6]
                     - [:3] = force (Newtons)
                     - [3:6] = torque (Newton-meters)
    """

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # ============================================================================
    # Linear Control (Force)
    # ============================================================================
    # Equation: F = Kp * error_x - Kd * v
    # This acts like a spring-damper system for each Cartesian direction (x, y, z)
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = (
        task_prop_gains[:, 0:3] * lin_error
        + task_deriv_gains[:, 0:3] * (0.0 - fingertip_midpoint_linvel)
    )

    # ============================================================================
    # Rotational Control (Torque)
    # ============================================================================
    # Equation: τ = Kp_rot * error_rot - Kd_rot * ω
    # This acts like a rotational spring-damper system (roll, pitch, yaw)
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = (
        task_prop_gains[:, 3:6] * rot_error
        + task_deriv_gains[:, 3:6] * (0.0 - fingertip_midpoint_angvel)
    )

    return task_wrench
