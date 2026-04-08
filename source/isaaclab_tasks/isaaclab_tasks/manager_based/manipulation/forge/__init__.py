# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FORGE (Force-guided Robotic GEometry manipulation) tasks.

This module provides manager-based implementations of the FORGE tasks:
- Peg Insert: Insert a peg into a hole
- Gear Mesh: Mesh gears together
- Nut Thread: Thread a nut onto a bolt

The tasks use differential inverse kinematics control with asset-relative actions
and multi-scale keypoint-based rewards for force-guided manipulation.
"""

# We leave this file empty to avoid importing configs that require pxr during package load.
# Environment registration is handled in config/franka/__init__.py
