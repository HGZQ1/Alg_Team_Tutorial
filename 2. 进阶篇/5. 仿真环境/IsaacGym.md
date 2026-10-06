# Isaac Gym

## 简介
**Isaac Gym 是 NVIDIA 出的一款 GPU 加速的机器人物理仿真器**，它是 NVIDIA Isaac 系列的一员，最大的特点是**把物理仿真和强化学习训练全部放在 GPU 上跑**，不需要 CPU 参与中间环节。

传统仿真器（比如 Gazebo、MuJoCo）在训练强化学习策略时，通常是 CPU 跑物理、GPU 跑神经网络，两边数据来回搬运，效率很低。而 Isaac Gym 直接在 GPU 上并行仿真**几千个环境**（比如同时跑 4096 个机器人），训练速度可以快一到两个数量级，这就是它出圈的原因——著名 的 **legged gym（四足机器人运动控制）** 和 **ANYmal Parkour** 等工作都是基于它做的。

### 优点
- GPU 并行仿真，训练速度极快，特别适合强化学习
- 内置 PhysX 物理引擎，对刚体、关节、接触的模拟效果不错
- 自带可视化，支持远程（headless）训练

### 缺点
- **NVIDIA 已经停止维护了！** 官方把重心转移到了继任者 Isaac Lab（基于 Isaac Sim）上


> 虽然已经停止更新，但目前大量足式机器人的开源项目（legged_gym、 IsaakRL 等）依然基于 Isaac Gym，学习它仍有很高的性价比，而且它的 API 比 Isaac Lab 简单很多，适合入门 GPU 并行仿真。

## 环境配置

### 前置要求
- NVIDIA GPU（建议显存 ≥ 8GB，RTX 30/40 系列最佳）
- NVIDIA 驱动 ≥ 525.60（Linux）
- Ubuntu 20.04 / 22.04
- Python 3.8（官方推荐，不建议用太新的 Python）

### 1. 创建虚拟环境
建议用 conda 隔离环境，Isaac Gym 对 Python 版本很挑剔：

```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
```

### 2. 下载 Isaac Gym Preview
官方下载页（需要注册 NVIDIA 账号）：
> https://developer.nvidia.com/isaac-gym

下载得到 `IsaacGym_Preview_4_Package.tar.gz`，解压：

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym
```

### 3. 安装依赖

```bash
# 先确认 CUDA 能用
nvidia-smi

# 安装 isaacgym 的 python 包
cd isaacgym/python
pip install -e .

# 安装 torch（版本要和 Isaac Gym 自带的 CUDA 版本匹配，这里以 cu113 为例）
pip install torch==1.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4. 验证安装

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

能弹出窗口看到一堆球掉下来就说明装好了。如果报错，常见问题：

- `ImportError: libpython3.8.so` → `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib`
- 闪退 / 段错误 → 大概率是显卡驱动太老，`nvidia-smi` 确认驱动版本
- 无显示器（服务器）→ 用 headless 模式，创建环境时传 `headless=True`

### 5.（可选）安装 legged_gym 生态

```bash
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
pip install -e .
```

跑起来：

```bash
python scripts/train.py --task=anymal_b_flat
```

能看到 4096 个环境同时训练，这就是 Isaac Gym 的魅力。

## 参考资料
- 官方文档：解压目录下的 `isaacgym/docs/index.html`
- legged_gym：https://github.com/leggedrobotics/legged_gym
