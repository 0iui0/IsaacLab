# FORGE 完整调用链路详解

本文档从强化学习的核心步骤角度，详细说明 FORGE 训练的完整调用链路，包含具体的代码文件和行号。

## 📋 目录

- [1. 初始化阶段](#1-初始化阶段)
- [2. 主训练循环](#2-主训练循环)
- [3. 数据收集](#3-数据收集)
- [4. GAE 优势函数计算](#4-gae-优势函数计算)
- [5. 数据集准备](#5-数据集准备)
- [6. PPO 梯度更新](#6-ppo-梯度更新)
- [7. 神经网络前向传播](#7-神经网络前向传播)

---

## 1. 初始化阶段

### 入口点

**文件**: `scripts/reinforcement_learning/rl_games/train.py`

```python
# L99-100: Hydra 加载配置
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """
    加载两个配置：
    - env_cfg: ForgeEnvCfg (环境配置)
    - agent_cfg: rl_games_ppo_cfg.yaml (RL 算法配置)
    """
```

### 环境创建

```python
# L182: 创建 Gym 环境
env = gym.make("Isaac-Forge-PegInsert-Direct-v0", cfg=env_cfg)
    └─> ForgeEnv.__init__()  (forge_env.py:L21-58)
        └─> FactoryEnv.__init__()  (factory_env.py:L26-39)
            └─> DirectRLEnv.__init__()  (direct_rl_env.py:L113-246)
```

### 环境包装

```python
# L203: 包装为 RL-Games 兼容环境
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    └─> RlGamesVecEnvWrapper.__init__()  (isaaclab_rl/rl_games/rl_games.py:L86-108)
```

### Runner 创建

```python
# L220-222: 创建 RL-Games Runner
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
    └─> 创建 A2CAgent  (a2c_continuous.py:L18-58)
        └─> 构建神经网络  (network_builder.py:L350-450)
```

---

## 2. 主训练循环

**文件**: `rl_games/common/a2c_common.py:L1046-1145`

```python
def train(self):
    """主训练循环"""
    self.init_tensors()  # L1047
    self.obs = self.env_reset()  # L1051

    while True:  # L1054
        epoch_num = self.update_epoch()  # L1055

        # 训练一个 epoch
        step_time, play_time, update_time, sum_time, \
        a_losses, c_losses, entropies, kls, last_lr, lr_mul = \
            self.train_epoch()  # L1056

        # 记录统计信息
        self.frame += curr_frames  # L1063
        self.write_stats(...)  # L1087

        # 保存模型
        if epoch_num % save_freq == 0:  # L1108
            self.save(path)

        # 检查退出条件
        if epoch_num >= self.max_epochs:  # L1115
            return self.last_mean_rewards, epoch_num
```

---

## 3. 数据收集

**文件**: `rl_games/common/a2c_common.py:L830-928`

```python
def play_steps_rnn(self):
    """收集 128 步交互经验"""

    for n in range(self.horizon_length):  # L832: 128 步

        # 3.1 保存 RNN 状态
        if n % self.seq_length == 0:  # L833: 每 128 步
            mb_rnn_states[:, n//seq_length] = self.rnn_states

        # 3.2 策略网络前向传播 → 获取动作
        res_dict = self.get_action_values(self.obs)  # L845
            └─> self.model.forward(obs)  (network_builder.py:L193)
                - LSTM(2层, 1024隐藏单元)
                - MLP(512→128→64)
                - Actor: mu, sigma (各7维)
                - Critic: value (1维)

        # 3.3 环境交互
        self.obs, rewards, self.dones, infos = \
            self.env_step(res_dict['actions'])  # L858
            └─> DirectRLEnv.step()  (direct_rl_env.py:L331-416)

        # 3.4 保存经验到 buffer
        self.experience_buffer.update_data('obses', n, self.obs['obs'])
        self.experience_buffer.update_data('actions', n, res_dict['actions'])
        self.experience_buffer.update_data('rewards', n, shaped_rewards)

        # 3.5 处理环境重置
        if len(all_done_indices) > 0:  # L876
            for s in self.rnn_states:
                s[:, all_done_indices, :] = 0.0

    # 3.6 计算 GAE 优势函数
    last_values = self.get_values(self.obs)  # L917
    mb_advs = self.discount_values(...)  # L921
    mb_returns = mb_advs + mb_values  # L922

    return batch_dict
```

### 环境交互详解

**文件**: `isaaclab/envs/direct_rl_env.py:L331-416`

```python
def step(self, action):
    """环境单步交互"""

    # 预处理动作
    action = action.to(self.device)  # L355
    if self.cfg.action_noise_model:
        action = self._action_noise_model(action)  # L357

    # 物理前处理
    self._pre_physics_step(action)  # L361
        └─> ForgeEnv._apply_action()  (forge_env.py:L144-229)

    # 物理仿真循环
    for _ in range(self.cfg.decimation):  # L368
        self._apply_action()  # L371
        self.scene.write_data_to_sim()  # L373
        self.sim.step(render=False)  # L375
        self.scene.update(dt=self.physics_dt)  # L382

    # 计算奖励
    self.reward_buf = self._get_rewards()  # L391
        └─> ForgeEnv._get_rewards()  (forge_env.py:L227-278)

    # 检查终止条件
    self.reset_terminated, self.reset_time_outs = self._get_dones()  # L389

    # 重置终止环境
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)  # L396

    # 获取观测
    self.obs_buf = self._get_observations()  # L408
        └─> ForgeEnv._get_observations()  (forge_env.py:L116-162)

    return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras
```

---

## 4. GAE 优势函数计算

**文件**: `rl_games/common/a2c_common.py:L551-567`

```python
def discount_values(self, fdones, last_values, mb_fdones, mb_values, mb_rewards):
    """
    计算 GAE (Generalized Advantage Estimation) 优势函数

    参数:
        fdones: 当前时刻的 done 标志
        last_values: 最后时刻的价值估计
        mb_fdones: 所有时刻的 done 标志
        mb_values: 所有时刻的价值估计
        mb_rewards: 所有时刻的奖励

    返回:
        mb_advs: GAE 优势函数
    """
    lastgaelam = 0  # L552
    mb_advs = torch.zeros_like(mb_rewards)  # L553

    # 从后向前计算
    for t in reversed(range(self.horizon_length)):  # L554
        if t == self.horizon_length - 1:  # L555
            nextnonterminal = 1.0 - fdones
            nextvalues = last_values
        else:
            nextnonterminal = 1.0 - mb_fdones[t+1]
            nextvalues = mb_values[t+1]

        # TD 残差
        delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal \
                - mb_values[t]  # L562

        # GAE 累加
        mb_advs[t] = lastgaelam = \
            delta + self.gamma * self.tau * nextnonterminal * lastgaelam  # L563-564

    return mb_advs
```

**GAE 参数配置**:

```yaml
# rl_games_ppo_cfg.yaml
gamma: 0.995  # 折扣因子
tau: 0.95    # GAE 参数 (lambda)
```

---

## 5. 数据集准备

**文件**: `rl_games/common/a2c_common.py:L1274-1328`

```python
def prepare_dataset(self, batch_dict):
    """准备 PPO 训练数据集"""

    # 5.1 计算优势函数
    advantages = returns - values  # L1285

    # 5.2 归一化 value 和 advantage
    if self.normalize_value:  # L1287
        values = self.value_mean_std(values)
        returns = self.value_mean_std(returns)

    if self.normalize_advantage:  # L1293
        advantages = (advantages - advantages.mean()) / \
                     (advantages.std() + 1e-8)

    # 5.3 构建数据集字典
    dataset_dict = {
        'old_values': values,          # 旧的价值估计
        'old_logp_actions': neglogpacs,  # 旧的动作对数概率
        'advantages': advantages,      # GAE 优势函数
        'returns': returns,            # 折扣回报
        'actions': actions,            # 执行的动作
        'obs': obses,                  # 观测
        'mu': mus,                     # 均值
        'sigma': sigmas,               # 标准差
        'rnn_states': rnn_states,      # RNN 状态
    }

    self.dataset.update_values_dict(dataset_dict)
```

---

## 6. PPO 梯度更新

**文件**: `rl_games/algos_torch/a2c_continuous.py:L93-171`

```python
def calc_gradients(self, input_dict):
    """计算 PPO 梯度并更新网络"""

    # 6.1 提取数据
    obs_batch = input_dict['obs']
    actions_batch = input_dict['actions']
    advantage = input_dict['advantages']
    old_action_log_probs_batch = input_dict['old_logp_actions']
    value_preds_batch = input_dict['old_values']
    return_batch = input_dict['returns']

    # 6.2 策略网络前向传播
    with torch.cuda.amp.autocast(enabled=self.mixed_precision):  # L117
        res_dict = self.model(batch_dict)  # L121
        action_log_probs = res_dict['prev_neglogp']
        values = res_dict['values']
        entropy = res_dict['entropy']
        mu = res_dict['mus']
        sigma = res_dict['sigmas']

    # 6.3 计算 Actor Loss (PPO-Clip)
    a_loss = self.actor_loss_func(
        old_action_log_probs_batch,
        action_log_probs,
        advantage,
        self.ppo,  # True
        curr_e_clip  # 0.2
    )  # L124-127

    # 6.4 计算 Critic Loss (Value Function)
    if self.has_value_loss:  # L129
        c_loss = common_losses.critic_loss(
            self.model,
            value_preds_batch,
            values,
            curr_e_clip,
            return_batch
        )  # L129-132

    # 6.5 计算 Entropy Bonus (探索激励)
    entropy = res_dict['entropy']  # L125

    # 6.6 计算 Bound Loss (动作范围约束)
    b_loss = self.bound_loss(mu)  # L138

    # 6.7 总 Loss
    loss = a_loss + 0.5 * c_loss * self.critic_coef \
            - entropy * self.entropy_coef \
            + b_loss * self.bounds_loss_coef  # L142

    # 6.8 反向传播
    for param in self.model.parameters():  # L147
        param.grad = None

    self.scaler.scale(loss).backward()  # L150

    # 6.9 梯度裁剪 & 优化器更新
    self.truncate_gradients_and_step()  # L152

    # 6.10 计算 KL 散度（用于自适应学习率）
    kl_dist = torch_ext.policy_kl(
        mu.detach(), sigma.detach(),
        old_mu_batch, old_sigma_batch
    )  # L155-159

    return a_loss, c_loss, entropy, kl_dist, ...
```

---

## 7. 神经网络前向传播

**文件**: `rl_games/algos_torch/network_builder.py:L193-332`

```python
def forward(self, obs_dict):
    """神经网络前向传播"""

    obs = obs_dict['obs']  # [batch, obs_dim]
    states = obs_dict['rnn_states']  # LSTM 状态
    dones = obs_dict.get('dones', None)

    # 7.1 LSTM 层
    if self.has_rnn:  # L220
        batch_size = obs.size(0)
        num_seqs = batch_size // seq_length  # L225
        out = out.reshape(num_seqs, seq_length, -1)  # L227
        out = out.transpose(0, 1)  # L229

        # LSTM 前向传播
        out, states = self.rnn(out, states, dones, bptt_len)  # L231
        # LSTM: input_size=obs_dim, hidden_size=1024, num_layers=2

        out = out.transpose(0, 1).reshape(batch_size, -1)  # L233

    # 7.2 MLP 层
    out = self.actor_mlp(out)  # L264
    # Sequential(
    #   Linear(1024, 512), ELU(),
    #   Linear(512, 128), ELU(),
    #   Linear(128, 64), ELU()
    # )

    # 7.3 Actor Head (策略网络)
    mu = self.mu_act(self.mu(out))  # L318
        # Linear(64, 7) → mu (动作均值)

    sigma = self.sigma_act(self.sigma(out))  # L320
        # Linear(64, 7) → sigma (动作标准差)

    # 7.4 Critic Head (价值网络)
    value = self.value_act(self.value(out))  # L259
        # Linear(64, 1) → value (状态价值)

    return {
        'mus': mu,
        'sigmas': sigma,
        'values': value,
        'prev_neglogp': action_log_probs,
        'entropy': entropy,
        'rnn_states': states
    }
```

---

## 🔄 完整训练循环总结

```python
while True:  # 每个 epoch
    # ===== 数据收集阶段 =====
    for n in range(128):  # horizon_length
        # 1. 策略网络前向传播
        actions, value = policy.forward(obs)

        # 2. 环境交互
        obs', reward, done = env.step(actions)

        # 3. 存储经验
        buffer.store(obs, actions, reward, value, done)

    # ===== PPO 更新阶段 =====
    # 4. 计算 GAE 优势函数
    advantages = discount_with_gae(rewards, values, dones)
    returns = advantages + values

    # 5. 准备数据集
    dataset = prepare_dataset(obs, actions, advantages, returns)

    # 6. 多次更新
    for mini_ep in range(4):  # mini_epochs_num
        for minibatch in dataset:
            # 6.1 策略网络前向传播
            new_mu, new_sigma, new_value = model.forward(obs)

            # 6.2 计算 PPO Loss
            actor_loss = PPO_CLIP(old_logp, new_logp, advantage)
            critic_loss = MSE(new_value, returns)
            entropy_bonus = mean(new_policy.entropy())
            total_loss = actor_loss + 0.5*critic_loss - entropy_bonus

            # 6.3 反向传播 & 优化
            total_loss.backward()
            optimizer.step()

            # 6.4 更新学习率
            lr, entropy_coef = scheduler.update(kl)

    # 7. 保存模型
    if epoch % save_freq == 0:
        save_model()
```

---

## 📊 关键数据流图

```
策略网络输出 (7维)
    ↓
观测 → 环境交互 (128步)
    ↓
经验 Buffer
    ↓
GAE 优势函数计算
    ↓
数据集准备 (归一化)
    ↓
PPO 更新 (4个 mini_epoch)
    ↓
网络参数更新
```

这个完整的调用链路展示了从训练脚本启动到环境交互的整个流程！
