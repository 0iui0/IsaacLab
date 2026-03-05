# FORGE 强化学习教程

本教程目录包含 FORGE（Force-based Reward and Success-guided Environment）任务的完整技术文档。

## 📚 文档目录

### 1. [完整调用链路](./01_complete_call_chain.md)
- 从命令行到环境交互的完整流程
- 代码行号索引
- 关键方法调用关系

### 2. [继承关系详解](./02_inheritance_hierarchy.md)
- Gym → RL-Games → Isaac Lab → Factory → Forge
- 接口适配层详解
- 关键方法重写位置

### 3. [阻抗控制器原理](./03_impedance_control_theory.md)
- 任务空间阻抗控制基础
- 雅可比矩阵转置的物理意义
- PD 控制律设计

### 4. [轴孔装配控制策略](./04_assembly_control_strategy.md)
- 分阶段装配策略
- 参数调优方法
- 阻抗控制的优势

### 5. [成功预测机制](./05_success_prediction.md)
- FORGE 的核心创新
- 第 7 维动作的作用
- 早期终止策略

### 6. [力传感器集成](./06_force_sensor_integration.md)
- 力传感器数据处理
- EMA 平滑算法
- 观测空间设计

## 🎯 快速开始

如果你想了解 FORGE 的核心概念，建议按以下顺序阅读：

1. 先读 [继承关系详解](./02_inheritance_hierarchy.md) 理解架构
2. 再读 [阻抗控制器原理](./03_impedance_control_theory.md) 理解控制
3. 最后读 [轴孔装配控制策略](./04_assembly_control_strategy.md) 理解应用

## 📖 代码注释

本教程相关的代码文件已添加详细注释：

- `source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py`

## 🔗 相关资源

- [FORGE 论文](https://arxiv.org/abs/...)
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [RL-Games GitHub](https://github.com/isaac-sim/rl_games)
