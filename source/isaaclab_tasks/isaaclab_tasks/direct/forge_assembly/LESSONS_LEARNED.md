# UR10 Forge Assembly - 经验教训总结

## 1. Docker 容器配置

### 关键问题
- **代码挂载路径**: 容器工作目录是 `/workspace/isaaclab`，不是 `/workspace/IsaacLab`
- **GPU 配置**: 需要正确配置 `deploy.resources.reservations.devices` 以访问 GPU

### 解决方案
```yaml
volumes:
  - ~/IsaacLab/source:/workspace/isaaclab/source:rw  # 挂载到正确路径

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["all"]
          capabilities: ["gpu"]
```

## 2. UR10 IK 收敛问题

### 问题描述
IK 求解器在环境 reset 时经常失败，导致训练卡住或跳过太多环境。

### 根本原因
1. **初始姿态不合适**: UR10 的默认姿态导致大部分随机目标超出工作空间
2. **fixed_asset 位置不当**: 固定资产放置位置不在 UR10 最优工作范围内

### 解决方案
```python
# UR10 profile 调整
reset_arm_joint_pos=[0.0, -0.5, -0.5, -0.5, 0.0, 0.0]  # 手臂向前向下

# Task config 调整
hand_init_pos: list = [0.0, 0.0, 0.05]  # 增加 Z 高度
fixed_asset_offset = torch.tensor([0.0, 0.50, 0.0], device=self.device)  # 50cm 距离
```

### 经验教训
- UR10 作为 6-DOF 机器人，工作空间比 Franka (7-DOF) 更受限
- 需要仔细调整初始姿态和目标分布以确保 IK 可解
- 添加 `max_ik_attempts` 防止无限循环

## 3. Peg 可视化与碰撞检测

### 问题
- Peg 作为固定资产连接到 EE link，但不会响应碰撞
- 可视化 peg 只是视觉效果，没有物理属性

### 完美解决方案
1. 创建 peg 和 force_sensor 作为 Collision Prim（不是 Rigid Body）
2. 通过 USD 层级关系继承 EE link 的运动
3. Peg 有自己的 CollisionAPI 和 PhysicsMaterialAPI 用于碰撞检测

### 实现代码
```python
def _spawn_peg_visualization(self):
    """Spawn peg as collision prim (not rigid body) child of EE link."""
    # Important: Don't use RigidBodyPropertiesCfg for children of articulation links!
    # Multiple rigid bodies in a hierarchy cause PhysX errors.
    peg_cfg = sim_utils.CylinderCfg(
        radius=peg_radius,
        height=peg_height,
        # No rigid_props - parent EE link handles the physics
        collision_props=sim_utils.CollisionPropertiesCfg(...),
        physics_material=sim_utils.RigidBodyMaterialCfg(...),
        visual_material=sim_utils.PreviewSurfaceCfg(...),
    )
    peg_cfg.func(peg_prim_path, peg_cfg, translation=(0, 0, -peg_height/2))
```

### 经验教训
- **USD 层级关系**: 子 prim 自动继承父 prim 的运动，但碰撞由各自的 Collision API 处理
- **FixedJoint vs 层级关系**: 层级关系足够用于刚性连接，不需要显式 FixedJoint
- **Spawn 时机**: 必须在 `clone_environments()` 之后 spawn，因为需要机器人 prim 存在
- **PhysX 层级限制**: 不要在 Articulation 的子 link 下创建新的 Rigid Body，会导致
  "missing xformstack reset" 错误。使用 Collision-only 代替。

## 4. Gym 环境注册

### 问题
- 新注册的环境在训练时提示"Environment doesn't exist"
- 模块导入顺序问题导致注册未执行

### 解决方案
- 确保 `forge_assembly/__init__.py` 被导入
- 使用绝对路径导入：`from isaaclab_tasks.direct import forge_assembly`
- 验证注册：`gym.spec("Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0")`

### 经验教训
- Python 模块导入顺序很重要
- 使用 `isaaclab.sh -p` 运行脚本会自动设置正确的 PYTHONPATH
- 容器内代码挂载后需要验证模块是否可导入

## 5. 代码结构与最佳实践

### RobotProfile 抽象
```python
@configclass
class RobotProfile:
    robot: ArticulationCfg
    num_arm_joints: int          # 6 (UR10) or 7 (Franka)
    has_gripper: bool
    ee_body_name: str
    grasp_type: str              # "gripper" or "fixed_peg"
    # ... 其他参数
```

### 优势
- 机器人无关的环境逻辑
- 易于添加新机器人 (UR10, CR5, etc.)
- 配置集中管理，便于调试

### 经验教训
- 避免硬编码 body names 和 joint indices
- 使用 profile 驱动所有机器人特定参数
- 为不同 grasp type 编写分支逻辑

## 6. 调试技巧

### 验证 Gym 注册
```bash
./isaaclab.sh -p -c "
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.direct import forge_assembly
for id in sorted(gym.registry.keys()):
    if 'ForgeAssembly' in id: print(id)
"
```

### 检查容器内代码挂载
```bash
docker exec <container> ls -la /workspace/isaaclab/source/.../forge_assembly/
```

### 训练快速测试
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-ForgeAssembly-UR10-PegInsert-Direct-v0 \
  --num_envs 8 --headless --max_iterations 5
```

## 7. 配置文件

完整的 docker-compose-forge-test.yml:
```yaml
services:
  isaac-lab-forge-test:
    image: nvcr.io/nvidia/isaac-lab:2.3.2
    container_name: isaac-lab-forge-test
    network_mode: host
    privileged: true
    volumes:
      - ~/IsaacLab/source:/workspace/isaaclab/source:rw
      # ... cache directories
    environment:
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
      - DISPLAY=${DISPLAY}
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["all"]
              capabilities: ["gpu"]
```

## 8. 关键教训总结

1. **路径一致性**: Docker 容器挂载路径必须与实际工作目录一致
2. **GPU 配置**: compose 文件需要正确配置 GPU 访问
3. **IK 调优**: 6-DOF 机器人需要更仔细的工作空间优化
4. **碰撞检测**: 可视化≠物理，需要正确的 USD 结构和 API
5. **模块导入**: Gym 注册依赖正确的模块导入顺序
6. **配置抽象**: RobotProfile 使代码更易维护和扩展
