# Isaac Lab

## 简介
**Isaac Lab（原名 Orbit）是 NVIDIA 基于 Isaac Sim 打造的开源机器人仿真框架**，官方定位就是 Isaac Gym 的继任者。Isaac Gym 停更之后，新的足式机器人强化学习工作（比如 extreme parkour 的一些复现、人形机器人训练）都逐渐迁移到了 Isaac Lab 上。

它和 Isaac Gym 的关系可以这么理解：

| | Isaac Gym | Isaac Lab |
| :--- | :--- | :--- |
| 底层 | 独立的轻量仿真器 | Isaac Sim（基于 Omniverse / USD） |
| 渲染 | 简陋 | 光追级渲染，接近真实照片 |
| 维护状态 | ❌ 停更 | ✅ 活跃更新 |
| 上手难度 | 较简单 | 概念多（USD、Omniverse），学习曲线陡 |
| 传感器仿真 | 基本没有 | 富传感器（相机、激光雷达、IMU 等）支持好 |

### 优点
- **官方长期维护**，是 NVIDIA 机器人仿真的主推方向
- 基于 PhysX 5，支持 GPU 并行物理，训练速度依然很快
- 传感器仿真强：可以直接拿到合成相机图像、深度图、点云，做视觉强化学习（sim-to-real 友好）
- 生态丰富：官方提供了机械臂操作、足式 locomotion、导航等大量示例任务

### 缺点
- 依赖 Isaac Sim（几个 GB 起步），安装包巨大，对磁盘和显卡要求高
- 抽象层次多（USD 场景描述、Omniverse 扩展），入门门槛比 Isaac Gym 高不少
- 只支持 NVIDIA GPU

## 环境配置

### 前置要求
- RTX 显卡（建议显存 ≥ 8GB，注意：**GTX 系列部分卡不被 Isaac Sim 支持**）
- NVIDIA 驱动 ≥ 525.60（Linux）
- Ubuntu 20.04 / 22.04
- 磁盘剩余空间 ≥ 50GB

### 方式一：pip 安装（推荐，最简单）

```bash
# 创建环境
conda create -n isaaclab python=3.10
conda activate isaaclab

# 装 PyTorch（去 pytorch.org 确认对应 CUDA 版本的命令，示例为 cu118）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 isaaclab
pip install isaaclab --no-deps
pip install isaaclab_task isaaclab_rl
```

### 方式二：源码安装（需要改源码 / 跑示例时用）

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

conda create -n isaaclab python=3.10
conda activate isaaclab

# 一键安装（自动装好 Isaac Sim 二进制和依赖，需要几十 GB 空间，耐心等待）
./isaaclab.sh --install
```

### 验证安装

```bash
# 打开示例场景，能看到一个地面 + 机器人的窗口就成功了
python scripts/tutorials/00_sim/create_empty.py
```

或者跑一个自带示例：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Velocity-Flat-Anymal-Direct-v0
```

### 常见问题
- 下载 Isaac Sim 很慢：可以用国内代理或者手动下载 `isaacsim` 的 pip 包
- 无显示器服务器：训练时加 `--headless`，可视化在本地用 livestream 或者录视频导出
- 显存不足报错：减少并行环境数量（`--num_envs 1024`）

## 参考资料
- 官方仓库：https://github.com/isaac-sim/IsaacLab
- 官方文档：https://isaac-sim.github.io/IsaacLab/
