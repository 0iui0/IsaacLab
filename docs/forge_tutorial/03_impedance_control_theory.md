# 阻抗控制器原理详解

本文档详细讲解任务空间阻抗控制的理论基础和实现细节。

## 📋 目录

- [1. 阻抗控制基础](#1-阻抗控制基础)
- [2. 任务空间 PD 控制](#2-任务空间-pd-控制)
- [3. 雅可比矩阵转置](#3-雅可比矩阵转置的物理意义)
- [4. 零空间控制](#4-零空间控制)
- [5. FORGE 的参数配置](#5-forge-的参数配置)

---

## 1. 阻抗控制基础

### 1.1 质量-弹簧-阻尼系统

```
二阶系统方程：

M·ẍ + D·ẋ + K·(x - x₀) = F_ext

其中：
- M：惯性矩阵 (质量)
- D：阻尼矩阵 (Damping)
- K：刚度矩阵 (Stiffness)
- x：当前位置
- x₀：平衡位置
- F_ext：外力 (接触力)
```

### 1.2 稳态响应（ẋ = 0, ẍ = 0）

```
K·(x - x₀) = F_ext

→ x = x₀ + F_ext / K

物理意义：
- 无接触（F_ext = 0）：x = x₀（到达目标）
- 有接触（F_ext ≠ 0）：x = x₀ + F_ext/K（柔性变形）
```

**关键洞察**：

```
刚度 K 的选择：

K 大（刚性）：
- x ≈ x₀（位置精确）
- F_ext = K·(x - x₀)（大力）
- 卡死风险高

K 小（柔性）：
- x = x₀ + F_ext/K（位置退让）
- F_ext 小（自动适应）
- 装配成功率高

→ 装配任务需要适中的 K！
```

---

## 2. 任务空间 PD 控制

### 2.1 PD 控制律

**文件**: `factory_control.py:L27-35`

```python
def _apply_task_space_gains(delta_fingertip_pose, linvel, angvel,
                            task_prop_gains, task_deriv_gains):
    """应用任务空间 PD 增益"""

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # 位置控制（线性）
    lin_error = delta_fingertip_pose[:, 0:3]  # 位置误差
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error \
                         + task_deriv_gains[:, 0:3] * (0.0 - linvel)
    # F = Kp * error - Kd * velocity

    # 旋转控制（角速度）
    rot_error = delta_fingertip_pose[:, 3:6]  # 旋转误差
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error \
                         + task_deriv_gains[:, 3:6] * (0.0 - angvel)
    # τ = Kp_rot * error - Kd_rot * angular_velocity

    return task_wrench
```

### 2.2 控制律详解

```
线性控制：
F_desired = Kp * (target_pos - current_pos) - Kd * current_vel

旋转控制：
τ_desired = Kp_rot * rot_error - Kd_rot * current_ang_vel
```

**参数含义**：

- `Kp`: 比例增益（弹簧系数）
- `Kd`: 导数增益（阻尼系数）
- `target_pos`: 目标位置
- `current_pos`: 当前位置
- `current_vel`: 当前速度

---

## 3. 雅可比矩阵转置的物理意义

### 3.1 雅可比矩阵定义

```
v = J · q̇

其中：
- v ∈ ℝ⁶：笛卡尔空间速度（线速度 + 角速度）
- q̇ ∈ ℝⁿ：关节空间速度
- J ∈ ℝ⁶ˣⁿ：雅可比矩阵

J = [∂x/∂q₁, ∂x/∂q₂, ..., ∂x/∂qₙ]
    第1列    第2列        第n列
```

### 3.2 虚功原理

```
功率守恒：

W = τᵀ · q̇ = Fᵀ · v

代入 v = J · q̇：

τᵀ · q̇ = Fᵀ · (J · q̇)
τᵀ · q̇ = (Jᵀ · F)ᵀ · q̇

对于任意 q̇：

τ = Jᵀ · F
```

**物理意义**：

```
雅可比矩阵 J 的列 = 每个关节单位速度对笛卡尔空间的影响

Jᵀ · F 的含义：
- F 在某个方向上"拉"末端
- Jᵀ 把这个"拉力"分配到各个关节
- 就像多个向量投影的叠加
```

### 3.3 代码实现

**文件**: `factory_control.py:L63-65`

```python
# 映射到关节空间
jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)
```

**具体例子**：

```
假设 2-DOF 平面机械臂

关节1（肩部）：角度 q₁，力矩 τ₁
关节2（肘部）：角度 q₂，力矩 τ₂
末端位置：(x, y)

雅可比矩阵：
J = [∂x/∂q₁, ∂x/∂q₂]
    [∂y/∂q₁, ∂y/∂q₂]

笛卡尔空间力：F = [Fx, Fy]ᵀ

关节力矩：
[τ₁]   [∂x/∂q₁, ∂y/∂q₁] [Fx]
[τ₂] = [∂x/∂q₂, ∂y/∂q₂] [Fy]

物理意义：
- τ₁ = (∂x/∂q₁)·Fx + (∂y/∂q₁)·Fy
  = 关节1运动对x/y的贡献 × 对应的力
- 同理 τ₂
```

---

## 4. 零空间控制

### 4.1 零空间的定义

```
任务空间（主任务）：
- 控制末端位置和姿态
- 优先级：高

零空间（次任务）：
- 不影响末端位形的关节运动
- 例如：姿态优化、避障、能量最小化
- 优先级：低
```

### 4.2 零空间投影矩阵

**文件**: `factory_control.py:L67-78`

```python
# 零空间投影矩阵
I = torch.eye(7, device=device).unsqueeze(0)
projection = (I - jacobian_T @ j_eef_inv)

# 零空间控制力
u_null = kp_null * (default_dof_pos - current_dof_pos) \
         - kd_null * current_dof_vel
torque_null = projection @ (arm_mass_matrix @ u_null)

# 总力矩 = 任务空间力矩 + 零空间力矩
dof_torque += torque_null
```

**数学原理**：

```
零空间投影：
P_null = I - Jᵀ · J⁺

其中：
- J⁺：雅可比伪逆
- P_null：投影到零空间的矩阵

性质：
- J · P_null = 0（零空间向量不影响末端）
- P_null² = P_null（投影算子）
```

**物理意义**：

```
零空间控制的作用：

1. 姿态优化
   - 优先级1：到达目标位置（任务空间）
   - 优先级2：接近默认姿态（零空间）

2. 避障
   - 优先级1：完成装配任务
   - 优先级2：远离障碍物

3. 能量最小化
   - 优先级1：跟踪轨迹
   - 优先级2：最小化关节力矩
```

---

## 5. FORGE 的参数配置

### 5.1 控制增益配置

**文件**: `forge_env_cfg.py:L21-28`

```python
@configclass
class ForgeCtrlCfg(CtrlCfg):
    # 位置增益（弹簧系数）Kp
    default_task_prop_gains = [565.0, 565.0, 565.0,   # Kp_x, Kp_y, Kp_z
                               28.0, 28.0, 28.0]      # Kp_roll, Kp_pitch, Kp_yaw

    # 导数增益通过临界阻尼计算
    # Kd = 2 * sqrt(Kp)
    # Kd_x = 2 * sqrt(565) ≈ 47.6 N·s/m
    # Kd_rot = 2 * sqrt(28) / rot_deriv_scale ≈ 1.06 Nm·s/rad
```

### 5.2 参数解读

```
位置刚度：Kp = 565 N/m
- 中等刚度：既有精度，又有柔性
- 5mm 的位置误差 → 2.8N 的力
- 通过裁剪和死区，实际力更小

旋转刚度：Kp_rot = 28 Nm/rad
- 较低的旋转刚度
- 允许姿态自适应调整
- 重要：轴孔装配时，姿态误差会导致卡死

阻尼系数：Kd = 2 * sqrt(Kp)
- 临界阻尼（无振荡）
- Kd_x = 47.6 N·s/m
- Kd_rot = 1.06 Nm·s/rad
```

### 5.3 动作空间边界

**文件**: `factory_env_cfg.py:L33-36`

```python
class CtrlCfg:
    # 动作空间边界
    pos_action_bounds = [0.05, 0.05, 0.05]     # ±5cm
    rot_action_bounds = [1.0, 1.0, 1.0]        # ±1 rad

    # 单步最大运动（防止过大动作）
    pos_action_threshold = [0.02, 0.02, 0.02]  # ±2cm/step
    rot_action_threshold = [0.097, ...]        # ±5.6°/step
```

---

## 6. 阻抗控制的优势

### 6.1 inherent compliance（内在柔顺性）

```python
# 不需要显式的力控制

传统力控制：
if force > threshold:
    adjust_position()  # 需要复杂的逻辑

阻抗控制：
F = Kp * (target - current)
# 当力 > threshold 时：
# Kp * (target - current) 产生柔顺力
# 因为 Kp 有限，不会无限推
```

### 6.2 自动适应不确定性

```python
# 装配误差自动补偿

场景：螺孔有轻微错位

刚性控制：
- 强行插入 → 卡死或损坏

阻抗控制：
- Kp * error → 产生接触力
- 力反馈 → 自动调整位置
- 螺母"滑"入正确位置
```

### 6.3 统一的位置和力控制

```python
# 远距离：位置控制主导
F ≈ Kp * large_displacement
velocity ↑, force ↓

# 接触后：力控制主导
F ≈ Kp * small_displacement
force ↑, 自动退让

# 无需切换控制模式！
```

---

## 7. 完整控制流程

```
1. 计算误差
   error = target_pose - current_pose

2. PD 控制律（笛卡尔空间）
   F = Kp * error - Kd * velocity
   τ = Kp_rot * rot_error - Kd_rot * angular_velocity

3. 死区处理
   F = apply_dead_zone(F, threshold)

4. 雅可比映射
   τ_joint = J^T · [F, τ]

5. 零空间控制
   τ_null = (I - J^T · J_inv) · u_null
   τ_total = τ_joint + τ_null

6. 限制力矩
   τ_total = clip(τ_total, -100, 100)
```

这种设计使得 FORGE 能够通过简单的阻抗控制完成复杂的轴孔装配任务！
