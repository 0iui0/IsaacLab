# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO agent configuration for Marvin M6 peg-in-hole assembly.

Uses recurrent PPO (LSTM) for force-feedback temporal reasoning.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class MarvinM6ForgeAssemblyRNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO training configuration for Marvin M6 peg-in-hole assembly."""

    num_steps_per_env = 512
    max_iterations = 3000
    save_interval = 100
    experiment_name = "forge_assembly_marvin_m6"
    empirical_normalization = False
    clip_actions = 1.0
    resume = False

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    # Recurrent policy with LSTM for force-feedback temporal reasoning
    policy = RslRlPpoActorCriticRecurrentCfg(
        state_dependent_std=True,
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        noise_std_type="log",
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
