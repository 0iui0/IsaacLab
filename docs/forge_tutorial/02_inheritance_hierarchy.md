# FORGE 继承关系详解

本文档详细说明从 Gym 标准接口到 FORGE 具体实现的完整继承层次，以及关键的调用位置。

## 📋 目录

- [1. 继承层次总览](#1-继承层次总览)
- [2. 各层级详解](#2-各层级详解)
- [3. 关键方法重写表](#3-关键方法重写表)
- [4. 完整调用链示例](#4-完整调用链示例)

---

## 1. 继承层次总览

```
Gym.Env (标准接口)
    ↓
rl_games.common.vecenv.IVecEnv (RL-Games 接口)
    ↓
RlGamesVecEnvWrapper (适配层)
    ↓
DirectRLEnv (Isaac Lab 基类)
    ↓
FactoryEnv (通用装配任务)
    ↓
ForgeEnv (FORGE 算法特化)
```

---

## 2. 各层级详解

### Layer 1: 标准接口层

#### Gym.Env (Gymnasium 标准)

**文件**: `gymnasium/envs/__init__.py`

```python
class gym.Env:
    """标准环境接口"""

    def reset(self, seed=None, options=None):
        """重置环境

        Returns:
            observation: 初始观测
            info: 额外信息
        """
        pass

    def step(self, action):
        """环境单步交互

        Args:
            action: 动作

        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止（成功/失败）
            truncated: 是否截断（超时）
            info: 额外信息
        """
        pass

    @property
    def observation_space(self):
        """观测空间"""
        pass

    @property
    def action_space(self):
        """动作空间"""
        pass
```

#### rl_games.common.vecenv.IVecEnv

**文件**: `rl_games/common/vecenv.py`

```python
class IVecEnv:
    """RL-Games 向量化环境接口"""

    def step(self, actions):
        """执行一步

        Args:
            actions: 动作张量 [num_envs, action_dim]

        Returns:
            obs_dict: {'obs': policy_obs, 'states': critic_obs}
            rewards: 奖励 [num_envs]
            dones: 终止标志 [num_envs]
            infos: 额外信息字典
        """
        pass

    def reset(self):
        """重置环境

        Returns:
            obs_dict: {'obs': policy_obs, 'states': critic_obs}
        """
        pass

    def get_env_info(self):
        """获取环境信息

        Returns:
            dict: {
                'observation_space': 观测空间,
                'action_space': 动作空间,
                'state_space': 状态空间 (Critic),
                'agents': 智能体数量
            }
        """
        pass
```

---

### Layer 2: RL-Games 适配层

#### RlGamesVecEnvWrapper

**文件**: `isaaclab_rl/rl_games/rl_games.py:L52-340`

```python
class RlGamesVecEnvWrapper(IVecEnv):
    """包装 Isaac Lab 环境以兼容 RL-Games"""

    def __init__(self, env, rl_device, clip_obs, clip_actions,
                 obs_groups=None, concate_obs_groups=True):
        """初始化包装器

        Args:
            env: Isaac Lab 环境 (DirectRLEnv 或 ManagerBasedRLEnv)
            rl_device: RL 训练设备 (例如 'cuda:0')
            clip_obs: 观测裁剪值
            clip_actions: 动作裁剪值
            obs_groups: 观测分组 (用于非对称 actor-critic)
            concate_obs_groups: 是否拼接观测组
        """
        # L118: 保存环境
        self.env = env

        # L119-123: 保存配置
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device

    def reset(self):  # L267-270
        """重置环境"""
        obs_dict, _ = self.env.reset()
        return self._process_obs(obs_dict)

    def step(self, actions):  # L272-298
        """执行一步"""
        # L274: 移动到 sim 设备
        actions = actions.detach().clone().to(device=self._sim_device)

        # L276: 裁剪动作
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        # L278: 调用底层环境 step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # L286: 处理观测
        obs_and_states = self._process_obs(obs_dict)

        # L289-293: 移动到 RL 设备
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)

        return obs_and_states, rew, dones, extras

    def _process_obs(self, obs_dict):  # L307-340
        """处理观测和状态

        功能:
        1. 移动到 RL 设备
        2. 裁剪观测
        3. 分离 actor 和 critic 观测
        """
        # L323-324: 移动到 RL 设备
        if self._rl_device != self._sim_device:
            obs_dict = {k: obs.to(device=self._rl_device) for k, v in obs_dict.items()}

        # L327-328: 裁剪观测
        for k, obs in obs_dict.items():
            obs_dict[k] = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # L331-338: 分离 actor/critic 观测
        rl_games_obs = {"obs": {group: obs_dict[group] for group in self._obs_groups["obs"]}}
        if len(self._obs_groups["states"]) > 0:
            rl_games_obs["states"] = {group: obs_dict[group] for group in self._obs_groups["states"]}

        return rl_games_obs
```

**关键特性**:

1. **设备管理**: 仿真在 GPU，训练在 GPU（可能不同）
2. **观测裁剪**: 防止数值溢出
3. **非对称 actor-critic**: Actor 用 policy_obs，Critic 用 state_obs

---

### Layer 3: Isaac Lab 环境基类

#### DirectRLEnv

**文件**: `isaaclab/envs/direct_rl_env.py:L53`

```python
class DirectRLEnv(gym.Env):
    """Direct RL 环境基类

    特点:
    - 手动实现 _get_observations, _apply_action, _get_rewards, _get_dones
    - 适合复杂的自定义逻辑 (如装配任务)
    """

    def __init__(self, cfg, render_mode=None, **kwargs):  # L113-246
        """初始化环境"""
        # L134-146: 创建仿真器
        self.sim = SimulationContext(
            render_mode=render_mode,
            physics_dt=cfg.sim.physics_dt,
        )

        # L148-157: 创建场景
        self.scene = Scene(cfg.scene)

        # L273-329: reset() 方法
        def reset(self, seed=None, options=None):
            """重置环境到初始状态"""
            indices = torch.arange(self.num_envs)
            self._reset_idx(indices)  # L313
            self.sim.forward()  # L317
            return self._get_observations(), self.extras  # L329

    def step(self, action):  # L331-416
        """环境单步交互 - 核心方法"""
        # L355-358: 预处理动作
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # L361: 物理前处理
        self._pre_physics_step(action)

        # L368-382: 物理仿真循环
        for _ in range(self.cfg.decimation):
            self._apply_action()  # L371 → 子类重写
            self.scene.write_data_to_sim()  # L373
            self.sim.step(render=False)  # L375 → PhysX 仿真
            self.scene.update(dt=self.physics_dt)  # L382

        # L389: 检查终止条件
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        # L391: 计算奖励
        self.reward_buf = self._get_rewards()

        # L396: 重置终止环境
        reset_env_ids = self.reset_buf.nonzero()
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # L408: 获取观测
        self.obs_buf = self._get_observations()

        # L416: 返回
        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras

    def _apply_action(self):  # L638-646
        """应用动作到环境 - 子类重写"""
        pass  # 基类为空，子类必须实现

    def _get_observations(self):  # L647-667
        """获取观测 - 子类重写"""
        pass

    def _get_rewards(self):  # L668-676
        """计算奖励 - 子类重写"""
        pass

    def _get_dones(self):  # L677-685
        """获取终止标志 - 子类重写"""
        pass
```

---

### Layer 4: 通用装配任务基类

#### FactoryEnv

**文件**: `isaaclab_tasks/direct/factory/factory_env.py:L23`

```python
class FactoryEnv(DirectRLEnv):
    """通用装配任务环境基类

    功能:
    - OSC (操作空间控制)
    - 任务空间 PD 控制
    - 装配任务奖励函数
    """

    def __init__(self, cfg, render_mode=None, **kwargs):  # L26-39
        """初始化装配环境"""
        # L27: 调用父类
        super().__init__(cfg, render_mode, **kwargs)

        # L36: 设置物体惯量
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)

        # L37: 初始化张量
        self._init_tensors()

        # L38: 设置默认动态参数
        self._set_default_dynamics_parameters()

    def _setup_scene(self):  # L85-127
        """设置场景 - 创建机器人和物体"""
        # L87: 生成地面
        spawn_ground_plane(...)

        # L89-93: 生成桌子
        spawn_table(...)

        # L95: 创建机器人
        self._robot = Articulation(self.cfg.robot)

        # L96: 创建手持物体 (held_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        # L97: 创建固定物体 (fixed_asset)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)

    def _apply_action(self):  # L253-311
        """应用动作 - OSC 控制

        FORGE 重写此方法实现资产相对动作空间
        """
        # L254-256: 获取动作
        pos_actions = self.actions[:, 0:3]
        rot_actions = self.actions[:, 3:6]

        # L258-285: 计算目标姿态
        ctrl_target_pos = current_pos + pos_actions
        ctrl_target_quat = compute_from_rot_actions(rot_actions)

        # L287-303: 裁剪动作
        delta_pos = ctrl_target_pos - current_pos
        pos_error_clipped = clip(delta_pos, -threshold, threshold)
        ctrl_target_pos = current_pos + pos_error_clipped

        # L305-311: 生成控制信号
        self.generate_ctrl_signals(ctrl_target_pos, ctrl_target_quat, 0.0)
            └─> factory_control.compute_dof_torque(...)
                # 任务空间 PD 控制
                # 雅可比映射

    def _get_observations(self):  # L194-251
        """获取观测"""
        # L195-235: 构建观测字典
        obs_dict = {
            'fingertip_pos_rel_fixed': self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            'fingertip_quat': self.fingertip_midpoint_quat,
            'ee_linvel': self.fingertip_midpoint_linvel,
            'ee_angvel': self.fingertip_midpoint_angvel,
        }

        # L238-250: 拼接张量
        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order)
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order)

        return {"policy": obs_tensors, "critic": state_tensors}

    def _get_rewards(self):  # L405-542
        """计算奖励"""
        # L415-439: 距离奖励
        dist_to_target = norm(held_pos - target_pos)
        reward_dist = -dist_to_target

        # L441-477: 旋转奖励
        rot_error = compute_rot_error(held_quat, target_quat)
        reward_rot = -rot_error

        # L479-493: 动作惩罚
        reward_action = -norm(delta_action)

        # L541: 返回总奖励
        return reward_dist + reward_rot + reward_action

    def _get_dones(self):  # L333-403
        """检查终止条件"""
        # L345-351: 检查任务成功
        success = check_task_success()

        # L353-355: 检查超时
        timed_out = episode_length > max_length

        return success, timed_out

    def _reset_idx(self, env_ids):  # L165-191
        """重置指定环境"""
        # L170-178: 重置机器人位置
        self._robot.reset(env_ids)

        # L180-189: 重置物体位置
        self._held_asset.reset(env_ids)
        self._fixed_asset.reset(env_ids)
```

---

### Layer 5: FORGE 算法特化

#### ForgeEnv

**文件**: `isaaclab_tasks/direct/forge/forge_env.py:L19`

```python
class ForgeEnv(FactoryEnv):
    """FORGE 算法特化环境

    核心创新:
    1. 力传感器集成
    2. 成功预测奖励
    3. 资产相对动作空间
    4. 早期终止策略
    """

    def __init__(self, cfg, render_mode=None, **kwargs):  # L21-58
        """初始化 FORGE 环境"""
        # L22: 调用父类
        super().__init__(cfg, render_mode, **kwargs)

        # L27-29: 初始化成功预测
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {
            thresh: torch.zeros(self.num_envs)
            for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]
        }

        # L43-51: 初始化力传感器
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6))

        # L53-60: 初始化动态参数
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains)
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold)
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold)

    def _compute_intermediate_values(self, dt):  # L60-115
        """计算中间值 (每帧调用)

        功能:
        1. 添加观测噪声
        2. 力传感器平滑
        3. 速度计算 (有限差分)
        """
        # L62: 调用父类
        super()._compute_intermediate_values(dt)

        # L65-92: 添加观测噪声
        pos_noise = torch.randn(...) * self.cfg.obs_rand.fingertip_pos
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + pos_noise

        rot_noise = torch.randn(...) * np.deg2rad(self.cfg.obs_rand.fingertip_rot_deg)
        self.noisy_fingertip_quat = quat_mul(self.fingertip_midpoint_quat, rot_noise)

        # L94-108: 力传感器平滑 (EMA)
        alpha = self.cfg.ft_smoothing_factor  # L95: 0.25
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + \
                                       (1 - alpha) * self.force_sensor_world_smooth  # L97

        # L110-115: 变换到指端坐标系
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = \
            forge_utils.change_FT_frame(...)

    def _get_observations(self):  # L116-162
        """获取观测 - 添加力传感器"""
        # L117: 调用父类
        obs_dict, state_dict = super()._get_observations()

        # L119-131: 添加 FORGE 特有观测
        obs_dict["ft_force"] = self.force_sensor_smooth[:, 0:3]  # 只用力，不用扭矩
        obs_dict["force_threshold"] = self.contact_penalty_thresholds

        # L162: 返回
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):  # L144-229
        """应用动作 - 资产相对动作空间"""
        # L143-149: 缩放动作
        pos_actions = self.actions[:, 0:3] @ diag(pos_action_bounds)
        rot_actions = self.actions[:, 3:6] @ diag(rot_action_bounds)

        # L151-181: 计算相对于 fixed_asset 的目标
        fixed_pos_action_frame = self.fixed_pos_obs_frame + noise
        ctrl_target_pos = fixed_pos_action_frame + pos_actions

        # L183-226: 裁剪动作
        delta_pos = ctrl_target_pos - current_pos
        pos_error_clipped = clip(delta_pos, -threshold, threshold)
        ctrl_target_pos = current_pos + pos_error_clipped

        # L228: 调用父类生成控制信号
        self.generate_ctrl_signals(ctrl_target_pos, ctrl_target_quat, 0.0)

    def _get_rewards(self):  # L227-278
        """计算奖励 - 添加成功预测"""
        # L228: 调用父类
        rew_buf = super()._get_rewards()

        # L246: 动作惩罚 (资产相对)
        pos_error = norm(delta_pos) / pos_threshold[0]
        rot_error = abs(delta_yaw) / rot_threshold[0]
        rew_dict["action_penalty_asset"] = pos_error + rot_error

        # L247-251: 接触惩罚
        contact_force = norm(force_sensor_smooth[:, 0:3])
        contact_penalty = relu(contact_force - threshold)
        rew_dict["contact_penalty"] = contact_penalty

        # L254-277: 成功预测奖励 (FORGE 核心创新)
        true_successes = self._get_curr_successes()
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # 第7维
        success_pred_error = abs(true_successes - policy_success_pred)

        # L275-276: 延迟启用
        if true_successes.mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        rew_dict["success_pred_error"] = success_pred_error * self.success_pred_scale

    def _reset_idx(self, env_ids):  # L346-383
        """重置环境 - 动态参数随机化"""
        # L347: 调用父类
        super()._reset_idx(env_ids)

        # L361-382: 随机化动态参数
        prop_gains = get_random_prop_gains(...)
        pos_threshold = get_random_prop_gains(...)
        rot_threshold = get_random_prop_gains(...)
        contact_penalty_thresholds = random(...)
        dead_zone_thresholds = random(...)
```

---

## 3. 关键方法重写表

| 方法 | DirectRLEnv | FactoryEnv | ForgeEnv | 作用 |
|------|-------------|------------|----------|------|
| `__init__` | L113-246 | L26-39 | L21-58 | 初始化 |
| `step` | L331-416 | 继承 | 继承 | 主循环 |
| `reset` | L273-329 | 继承 | 继承 | 重置环境 |
| `_apply_action` | L638-646 (空) | L253-311 | L144-229 | 应用动作 |
| `_get_observations` | L647-667 | L194-251 | L116-162 | 获取观测 |
| `_get_rewards` | L668-676 | L405-542 | L227-278 | 计算奖励 |
| `_get_dones` | L677-685 | L333-403 | 继承 | 检查终止 |
| `_reset_idx` | - | L165-191 | L346-383 | 重置环境 |

---

## 4. 完整调用链示例

### 示例：从训练脚本到环境交互

```python
# 1. 训练脚本
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Forge-PegInsert-Direct-v0 --num_envs=1

# 2. Hydra 加载配置
@hydra_task_config(args_cli.task, args_cli.agent)
main(env_cfg, agent_cfg)
    ├── env_cfg: ForgeEnvCfg
    └── agent_cfg: rl_games_ppo_cfg.yaml

# 3. 创建环境
env = gym.make("Isaac-Forge-PegInsert-Direct-v0", cfg=env_cfg)
    └─> ForgeEnv.__init__(cfg)
        └─> FactoryEnv.__init__(cfg)
            └─> DirectRLEnv.__init__(cfg)

# 4. 包装环境
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

# 5. 创建 Runner
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
    └─> A2CAgent.__init__(params)
        └─> self.model = self.network.build(build_config)

# 6. 开始训练
runner.run({"train": True})
    └─> A2CAgent.train()
        └─> while True:
            train_epoch()
                └─> play_steps_rnn()
                    └─> env_step(actions)
                        └─> DirectRLEnv.step(actions)
                            └─> ForgeEnv._apply_action()
                            └─> ForgeEnv._get_rewards()
                            └─> ForgeEnv._get_observations()
```

---

## 5. 数据流转图

```
RL-Games (A2CAgent)
    │
    ├─ play_steps_rnn()
    │   └─> self.env.step(actions)
    │       └─> RlGamesVecEnvWrapper.step()
    │           └─> DirectRLEnv.step()
    │               ├─> _pre_physics_step(action)
    │               │   └─> ForgeEnv._apply_action()
    │               ├─> sim.step()
    │               ├─> _get_rewards()
    │               │   └─> ForgeEnv._get_rewards()
    │               └─> _get_observations()
    │                   └─> ForgeEnv._get_observations()
    │
    └─> calc_gradients()
        └─> model.forward(obs)
            └─> Actor-Critic 网络
                ├─> LSTM
                ├─> MLP
                ├─> Actor Head (mu, sigma)
                └─> Critic Head (value)
```

---

## 6. 关键设计决策

### 为什么需要这么多层次？

1. **Gym.Env**: 标准接口，兼容所有 RL 库
2. **IVecEnv**: RL-Games 接口，支持批量训练
3. **RlGamesVecEnvWrapper**: 适配层，处理设备转换和裁剪
4. **DirectRLEnv**: Isaac Lab 基类，处理仿真循环
5. **FactoryEnv**: 通用装配逻辑，OSC 控制器
6. **ForgeEnv**: FORGE 特化，力传感器和成功预测

### 每一层的职责

| 层级 | 职责 | 关键功能 |
|------|------|----------|
| Gym | 标准化 | reset(), step() 接口 |
| IVecEnv | 批量化 | 并行环境训练 |
| Wrapper | 适配 | 设备转换，观测裁剪 |
| DirectRLEnv | 仿真 | 物理循环，场景管理 |
| FactoryEnv | 控制逻辑 | OSC，装配奖励 |
| ForgeEnv | 任务特化 | 力传感器，成功预测 |

这种分层设计使得代码清晰、可维护、可扩展！
