# MuJoCo

## 简介
**MuJoCo（Multi-Joint dynamics with Contact）是一个老牌的高性能物理仿真器**，最早由华盛顿大学 Roboti LLC 开发，主打**接触动力学的精度和速度**。2021 年 DeepMind 收购了它，2022 年起**完全开源免费**（Apache 2.0 协议），从此成为学术界最流行的机器人仿真器之一。

很多经典工作都基于 MuJoCo：OpenAI Gym 的经典控制任务、DeepMind 的 DM Control Suite、足式机器人领域的 **MJX（MuJoCo XLA）**、以及如今人形机器人领域很火的 **MuJoCo Playground / mjlab** 等。

### 优点
- **免费开源**，跨平台（Linux / Windows / macOS），**AMD 显卡和纯 CPU 也能跑**，不像 Isaac 系列绑死 NVIDIA
- 接触动力学精度高，关节/肌腱建模能力强，物理结果可信度高
- 轻量，安装只要 `pip install`，秒装好，对新手最友好
- 社区生态极其丰富：Gymnasium、DM Control、robosuite、MJX（GPU 加速版）都基于它
- 一般来说如果我们用Isaacgym/Isaaclab训练好的模型要在Mujoco跑一遍sim2sim看看效果，然后再虑部署到real

### 缺点
- 单线程仿真，单个 CPU 核跑一个环境，**原生不支持 GPU 并行多环境**（要 GPU 并行得用 MJX 或 Brax）
- 渲染效果一般（不过 3.x 版本已经改善很多）
- 模型格式是自家的 MJCF（XML），上手要学一套标签语法

> 补充：**MJX（MuJoCo XLA）** 是 MuJoCo 的 GPU 加速版本，用 JAX 把物理计算编译到 GPU 上并行跑几千个环境，定位类似 Isaac Gym，现在很多足式 RL 训练已经切换到 MJX 了。

## 环境配置

### 前置要求
- 啥都行：Linux / Windows / macOS，有无独显均可（MuJoCo 用 OpenGL 渲染，不依赖 CUDA）

### 1. 安装（pip 一行搞定）

```bash
conda create -n mujoco python=3.10
conda activate mujoco

pip install mujoco
```

### 2. 验证安装

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

能打印出版本号（如 `3.2.x`）就装好了。再跑个可视化试试：

```bash
pip install mujoco-python-viewer   # 或使用官方 viewer
python -m mujoco.viewer
```

会打开一个空窗口，把任意 MJCF 模型（XML 文件）拖进去就能看到 3D 模型，可以用滑块直接拖动关节。

### 3. 快速上手示例

```python
import mujoco

# 加载模型（xml 可以用官方模型库 mujoco_menagerie 里的）
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 仿真 1000 步
for _ in range(1000):
    mujoco.mj_step(model, data)
    print(data.qpos)  # 打印关节位置
```

### 4.（推荐）安装配套生态

```bash
# Gymnasium 的 mujoco 环境（经典 RL 任务集）
pip install gymnasium[mujoco]

# DeepMind 控制套件
pip install dm_control

# 官方维护的高质量机器人模型库（包含各种四足、机械臂、人形模型）
git clone https://github.com/google-deepmind/mujoco_menagerie
```

### 5.（可选）GPU 加速版 MJX

需要 NVIDIA / AMD 显卡 + JAX：

```bash
pip install "mujoco[mujoco-mjx]" "jax[cuda12]"
```

### 常见问题
- Linux 下渲染报错 / `GLFW` 相关错误：无显示器环境加 `export MUJOCO_GL=egl`（有 NVIDIA 卡）或 `export MUJOCO_GL=osmesa`（纯软件渲染）
- 找不到模型：推荐从 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 拿模型，都是官方整理好的

## 参考资料
- 官方文档：https://mujoco.readthedocs.io
- 官方模型库：https://github.com/google-deepmind/mujoco_menagerie
- MJX 论文：https://arxiv.org/abs/2403.19553
