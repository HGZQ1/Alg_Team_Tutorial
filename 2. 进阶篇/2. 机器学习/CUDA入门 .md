# CUDA入门

> 编号占位说明：本章内部编号统一写成 `X.1`、`X.2`……定好章号之后，把全文的 `X.` 全部替换成对应数字即可。

## 这一章是干什么的

**CUDA 是 NVIDIA 提供的一套东西，让我们能把计算任务交给显卡去做。**

深度学习的运算说白了就是海量的矩阵乘法，交给显卡跑通常比 CPU 快几十到上百倍。一个 YOLO 模型在 CPU 上推理一张图要几百毫秒，在显卡上可能只要一两毫秒——机器人能不能实时反应，差别就在这里。

我们平时不会直接写 CUDA 代码，PyTorch 和 Ultralytics 已经把它包好了。我们要做的只有两件事：**把环境装对**，以及**出问题时知道该查哪一层**。

所以这一章只讲三件事：

```text
   1. 那一堆版本号分别是什么（驱动 / CUDA / cuDNN / PyTorch）
   2. 怎么装、怎么验证装对了
   3. 报错和性能问题怎么排查
```

原理性的东西（显卡为什么快、kernel 怎么调度、怎么手写算子）本章只做最简单的交代——硬件层面的差异 3.0 节已经讲过了，想深入的看章末参考资料。

## 本章学习目标

完成本章后，你应当能够：

- 分清 **NVIDIA 驱动**、**CUDA Toolkit**、**cuDNN**、**PyTorch 自带的 CUDA 运行时**这四者的关系；
- 看懂 `nvidia-smi` 的输出，特别是右上角那个 CUDA 版本号的真实含义；
- 独立完成驱动和 PyTorch 的安装，并跑通一份验证脚本；
- 查出自己显卡的**计算能力（sm_xx）**，据此判断该装哪个版本；
- 在 PyTorch 里正确地把模型和数据放上 GPU；
- 测出一段代码在 GPU 上的真实耗时；
- 遇到常见的 CUDA 报错时，知道从哪一层开始查。

---

## X.1 显卡和 CPU 的分工（简版）

3.0 节讲过硬件，这里只补和我们直接相关的一句：

**CPU 是十几个很聪明的核心，GPU 是几千个很笨的核心。**

```text
   CPU：核心少，每个都能处理复杂逻辑     → 适合有分支、有依赖的任务
   GPU：核心多，每个只会做简单算术       → 适合"同一件简单事重复几千遍"
```

深度学习恰好是后者——矩阵乘法里每个乘加都独立、都简单，几千个核心一起上，效率就出来了。

由此推出一条实用结论：**不是什么东西丢给 GPU 都会变快。** GPU 有启动成本，数据要从内存搬到显存、算完再搬回来。对一张 640×480 的小图做几次形态学操作，搬运时间比计算时间还长，用 GPU 反而更慢。

| 任务 | 适合 GPU 吗 |
| --- | --- |
| 神经网络推理 / 训练 | 非常适合 |
| 大规模图像卷积、点云处理 | 适合 |
| 5 章那个色块识别 | 通常不适合，图太小 |
| 读文件、解析配置 | 不适合 |

> 有个名词得提一句，因为报错里会反复出现：**kernel（核函数）指的是一段"要在 GPU 上被几千个线程同时执行"的代码**。我们不需要自己写，PyTorch 里已经有成千上万个写好的。知道这个词的意思，是为了看懂 `no kernel image is available` 这类报错——它说的是"这个包里没有能在你这块显卡上跑的二进制代码"，详见 X.4。

---

## X.2 把那一堆名词理清楚

这一节可能是整章最有价值的部分。新生 90% 的困惑来自于分不清下面这几个东西。

### X.2.1 四个层次

```text
   ┌──────────────────────────────────────────────┐
   │  你的代码：PyTorch / Ultralytics / OpenCV     │   ← 你写的
   ├──────────────────────────────────────────────┤
   │  加速库：cuDNN / cuBLAS / TensorRT            │   ← 别人写好的算子
   ├──────────────────────────────────────────────┤
   │  CUDA Runtime / Toolkit                      │   ← 编译和运行的工具链
   ├──────────────────────────────────────────────┤
   │  NVIDIA 驱动 (Driver)                        │   ← 和内核打交道
   ├──────────────────────────────────────────────┤
   │  显卡硬件                                     │
   └──────────────────────────────────────────────┘
```

| 名词 | 是什么 | 装在哪 | 怎么查版本 |
| --- | --- | --- | --- |
| **NVIDIA 驱动** | 让操作系统认识显卡的底层软件 | **必须装在宿主机上** | `nvidia-smi` 左上角 |
| **CUDA Toolkit** | 开发套件，含编译器 `nvcc`、头文件、示例 | 宿主机或容器里都行 | `nvcc -V` |
| **CUDA Runtime** | 程序运行时真正调用的那套库（`libcudart.so`） | 跟着程序走 | 见下 |
| **cuDNN** | NVIDIA 写的深度学习算子库（卷积、池化等） | 跟着 PyTorch 走 | `torch.backends.cudnn.version()` |

### X.2.2 最重要的一条：你多半不需要装 CUDA Toolkit

这条能省你很多时间。

**用 pip 装的 PyTorch，包里自带了 CUDA Runtime 和 cuDNN。** 也就是说：

```bash
pip install torch
```

装下来那一两个 GB 里，除了 PyTorch 本身，还塞了一整套 CUDA 运行时库。你不需要在系统里另外装 CUDA Toolkit，`torch.cuda.is_available()` 照样是 `True`。

那什么时候才需要 Toolkit？

- 你要**编译**别人的 CUDA 源码（比如某些需要现场编译算子的库）；
- 你要自己写 `.cu` 文件；
- 你要用 `nvcc`、`nsight` 这些开发工具。

对纯做推理和训练的我们来说，**只要宿主机驱动够新，装个 pip 版 PyTorch 就能开工**。

> 6 章那份 Dockerfile 里装了 `cuda-toolkit-12-6`，是当时抱着"以后可能要编译点什么"的想法装的，代价是镜像胖了好几个 GB。如果你现在从头写一份只做推理的 Dockerfile，这一段完全可以省掉。

### X.2.3 `nvidia-smi` 右上角那个数字骗了很多人

在宿主机上敲：

```bash
nvidia-smi
```

输出大概长这样：

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.10    Driver Version: 570.86.10    CUDA Version: 12.8      |
|-----------------------------------------------+----------------------+------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile ...|
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================================+======================+======|
|   0  NVIDIA GeForce RTX 5080 Laptop      On   | 00000000:01:00.0  On |  N/A |
| N/A   45C    P8    12W /  175W |   1024MiB / 16384MiB |      3%      Default |
+-----------------------------------------------+----------------------+------+
```

**右上角的 `CUDA Version: 12.8` 不是"你装了 CUDA 12.8"。**

它的真正含义是：**这个驱动最高能支持到 CUDA 12.8。** 你实际用的可能是 12.1、11.8，都没问题，只要不超过这个上限。

所以会出现这种看起来矛盾、其实完全正常的情况：

```bash
nvidia-smi          # 显示 CUDA Version: 12.8
nvcc -V             # 显示 release 12.6
python3 -c "import torch; print(torch.version.cuda)"   # 显示 12.1
```

三个数字全不一样，但程序跑得好好的。因为它们说的是三件事：**驱动能支持到 12.8**、**装的开发工具是 12.6**、**PyTorch 用的运行时是 12.1**。

记住一条规则就够了：

> **驱动版本要 ≥ 程序需要的 CUDA 版本。** 驱动是向后兼容的，新驱动能跑老 CUDA 程序，反过来不行。所以驱动尽量装新的，CUDA 版本按框架要求来。

其余几个关键字段：

| 字段 | 含义 | 你该关注什么 |
| --- | --- | --- |
| `Pwr:Usage/Cap` | 当前功耗 / 上限 | 训练时如果远低于上限，可能被节能模式限制了 |
| `Memory-Usage` | 显存占用 / 总量 | 跑不动的时候先看这里 |
| `GPU-Util` | GPU 利用率 | 一直很低说明瓶颈不在 GPU |
| `Perf` | 性能状态 P0~P12 | P0 是全速，P8 是空闲 |

实时刷新：

```bash
watch -n 0.5 nvidia-smi
```

---

## X.3 安装与配置

这一节是本章的操作主体。顺序不能颠倒：**先驱动，再 PyTorch。**

### X.3.1 第一步：装 NVIDIA 驱动

驱动只能装在宿主机上，容器里装不了（6 章讲过原因：容器共用宿主机内核）。

Ubuntu 上最省事的方式：

```bash
# 看看系统推荐哪个版本
ubuntu-drivers devices

# 装推荐版本
sudo ubuntu-drivers autoinstall

# 重启
sudo reboot
```

重启后验证：

```bash
nvidia-smi
```

能打印出显卡信息就成了。**这一步不通，后面全都免谈**，先解决它再往下。

> 想手动装指定版本的话，官方驱动下载页在这里：
> <https://www.nvidia.cn/drivers/lookup/>
>
> 如果 `nvidia-smi` 报 `couldn't communicate with the NVIDIA driver`，通常是装完没重启，或者 BIOS 里的 Secure Boot 拦了。

### X.3.2 第二步：确认该装哪个版本

一般按显卡代数选就行：

| 你的显卡 | 装什么 |
| --- | --- |
| NVIDIA 30 / 40 系 | CUDA 12.6 版（`cu126`） |
| NVIDIA 50 系 | CUDA 12.8 版（`cu128`） |
| 没有 NVIDIA 显卡 | CPU 版 |
| Jetson | **不能用 pip 通用包**，要用 NVIDIA 为 JetPack 编的专用版 |

拿不准的时候，用 X.4 讲的方法查一下自己显卡的算力，再对着 PyTorch 官网的版本说明选。

### X.3.3 第三步：装 PyTorch

**安装命令一律去官网生成，不要抄旧教程里的。** 官方选择器会根据你选的系统、包管理器、CUDA 版本给出准确命令：

> **PyTorch 安装选择器**：<https://pytorch.org/get-started/locally/>
> **CUDA Toolkit 下载页**（确实需要 Toolkit 时才去）：<https://developer.nvidia.com/cuda-downloads>
> **CUDA 显卡算力对照表**：<https://developer.nvidia.com/cuda-gpus>

典型命令长这样：

```bash
# CUDA 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU 版
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**`--index-url` 不能省。** 不加这个参数，pip 默认装的多半不是你要的那个变体——这是最高频的踩坑点，每年都有人装成 CPU 版，然后找半天为什么用不了显卡。

装完先看版本字符串：

```python
import torch
print(torch.__version__)
```

```text
2.5.1+cu126      ← 配 CUDA 12.6，对了
2.5.1+cpu        ← CPU 版，没有 GPU 支持
```

**加号后面那截是关键。** 看到 `+cpu` 却发现 `is_available()` 是 `False`，那不是环境坏了，是包装错了。

### X.3.4 第四步：验证

把这段存成 `check_cuda.py`，以后每次装完环境都跑一遍：

```python
import torch

print("PyTorch      :", torch.__version__)
print("CUDA 可用    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("没有可用的 GPU。检查顺序：")
    print("  1. 版本字符串是不是 +cpu")
    print("  2. 宿主机 nvidia-smi 通不通")
    print("  3. 如果在容器里，有没有加 --gpus all / runtime: nvidia")
    raise SystemExit

print("显卡         :", torch.cuda.get_device_name())
print("算力         :", torch.cuda.get_device_capability())
print("torch 支持   :", torch.cuda.get_arch_list())
print("CUDA 运行时  :", torch.version.cuda)
print("cuDNN        :", torch.backends.cudnn.version())

# 关键：真的算一次
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
torch.cuda.synchronize()
print("矩阵乘法通过 :", y.shape)
```

**最后那三行不能省。** `is_available()` 返回 `True` 只说明 CUDA 初始化成功了，算力和版本对不上的话，真正执行算子的那一刻才会崩——必须真的算一次才算验证通过。

> 在容器里跑的话，还要先确认容器能看到显卡。6 章讲过：宿主机装 `nvidia-container-toolkit`，compose 里写 `runtime: nvidia`，然后
> `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
> 能出信息就通了。

---

## X.4 计算能力 sm_xx：装错版本的根源

### X.4.1 每一代显卡有一个"方言"

NVIDIA 每一代显卡的指令集都不完全一样，用**计算能力（Compute Capability）**来标识，写成 `sm_xx`：

| 架构 | 代表显卡 | 计算能力 |
| --- | --- | --- |
| Pascal | GTX 10 系 | sm_61 |
| Turing | RTX 20 系 | sm_75 |
| Ampere | RTX 30 系 | sm_86 |
| Ada Lovelace | RTX 40 系 | sm_89 |
| Blackwell | RTX 50 系 | **sm_120** |
| Ampere (嵌入式) | **Jetson Orin** | **sm_87** |

查自己的：

```python
import torch
print(torch.cuda.get_device_capability())   # 例如 (12, 0)，即 sm_120
print(torch.cuda.get_device_name())         # NVIDIA GeForce RTX 5080 Laptop GPU
```

### X.4.2 `no kernel image` 到底怎么回事

一个编译好的 PyTorch 包里，装的是**针对若干个特定 sm 编译好的二进制 kernel**。比如某个版本可能编译了 sm_75 / sm_80 / sm_86 / sm_89 四套。

如果你的显卡是 sm_120，而包里没有 sm_120 的版本，也没有能兼容的中间代码，运行时就找不到能跑的 kernel：

```text
no kernel image is available for execution on the device
```

或者更直白的：

```text
NVIDIA GeForce RTX 5080 Laptop GPU with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```

**最坑的地方是：`torch.cuda.is_available()` 返回 `True`。** 驱动认得显卡、CUDA 也初始化成功了，只有真正要执行算子的那一刻才崩。所以 X.3.4 的验证脚本里必须有那次真实的矩阵乘法。

正确的自检方式是**真的算一下**：

```python
import torch

print("可用:", torch.cuda.is_available())
print("显卡:", torch.cuda.get_device_name())
print("算力:", torch.cuda.get_device_capability())
print("torch 支持的架构:", torch.cuda.get_arch_list())

# 关键：真的做一次运算
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
torch.cuda.synchronize()
print("矩阵乘法通过，结果:", y.shape)
```

`get_arch_list()` 会打印这个 PyTorch 支持哪些架构，比如 `['sm_75', 'sm_80', 'sm_86', 'sm_90']`。**如果你的算力不在这个列表里，就是版本不对，换 PyTorch 版本，不要试图去改代码。**

> 这就是 6 章那个 `GPU_SERIES` 分支的由来：40 系（sm_89）用一套 whl，50 系（sm_120）用另一套。新硬件刚出来的那段时间，稳定版框架往往还没跟上，只能用预览版。

## X.5 在 PyTorch 里用 GPU

### X.5.1 设备的概念

PyTorch 里每个张量（Tensor）都有一个"住址"，要么在 CPU 内存，要么在某块 GPU 的显存里。

```python
import torch

# 标准写法：能用 GPU 就用，不能就退回 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x = torch.randn(3, 4)               # 默认在 CPU
x = x.to(device)                    # 搬到 GPU
print(x.device)                     # cuda:0

y = torch.randn(3, 4, device=device)  # 直接在 GPU 上创建（更好）
```

模型也一样：

```python
model = MyModel()
model = model.to(device)
```

> 上面两种建张量的写法有区别：`torch.randn(3,4).to(device)` 是先在 CPU 上造好再搬过去，`torch.randn(3,4, device=device)` 是直接在 GPU 上造。后者少一次搬运，能直接写就直接写。

### X.5.2 最常见的报错

```text
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

意思很直白：**你想让一个在显存里的张量和一个在内存里的张量一起做运算，GPU 够不着后者。**

```python
a = torch.randn(3, 4, device='cuda')
b = torch.randn(3, 4)                # 忘了搬
c = a + b                            # 报错

c = a + b.to('cuda')                 # 修好了
```

这个错在自己写推理流程时特别容易出现：模型搬上去了，但预处理出来的那个张量忘了搬。养成习惯——**数据进模型之前，最后一步永远是 `.to(device)`**。

### X.5.3 结果拿回 CPU

算完之后想看结果、想画框、想存文件，都得把数据搬回 CPU：

```python
result = output.cpu().numpy()
```

如果张量参与过反向传播，还要先断开梯度：

```python
result = output.detach().cpu().numpy()
```

顺序记成 **`detach` → `cpu` → `numpy`**，反过来会报错。

---

## X.6 测一下速度

改完东西到底快了没有，得测。测 GPU 有一个规则必须知道：

> **GPU 调用是异步的。** 你写 `y = model(x)`，CPU 只是把任务扔进队列就返回了，并不等 GPU 算完。所以直接用 `time.time()` 掐，测出来的是"提交任务"的时间，会得到 0.05ms 之类的荒谬结果。
>
> **计时前后都要加 `torch.cuda.synchronize()`**，强制等 GPU 干完活。

一个能直接用的模板：

```python
import time
import torch

def benchmark(model, x, warmup=20, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):        # 预热
            model(x)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(runs):
            model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0

    print(f"平均 {dt/runs*1000:.2f} ms  |  {runs/dt:.1f} FPS")
```

两个要点：

- **必须预热。** 前几次推理会慢很多（CUDA 上下文初始化、cuDNN 挑算法、显存分配），跑几十次热身之后测的才是稳态性能。反过来说，实际部署时机器人开机后最好先拿假图跑几遍预热，别等真要用时才第一次调用。
- **`.item()` / `.cpu()` / `print(tensor)` 会隐式同步。** 所以如果你发现耗时全堆在某一行 `.cpu()` 上，**别去优化那一行**——它在替前面所有异步操作背锅，问题在它前面。

还有一点：整体 FPS 低不代表模型慢。一个视觉流程有好几步，任何一步都可能是瓶颈：

```text
   读一帧 → 预处理 → 搬到GPU → 推理 → 搬回CPU → 后处理 → 显示
```

分段测经常有意外发现：推理只要 2ms，但读一帧要 28ms——那是相机只有 30fps，程序已经比相机快太多了，再优化 GPU 也没有收益。

**先测，再优化。** 凭感觉优化，十次有八次优化错了地方。

---

## X.7 常见问题排查表

| 现象 | 可能原因 | 怎么办 |
| --- | --- | --- |
| `nvidia-smi` 说找不到驱动 | 装完没重启 / Secure Boot 拦了 | 重启；BIOS 里关 Secure Boot |
| `torch.cuda.is_available()` 是 `False` | 装的是 CPU 版 PyTorch | 看 `torch.__version__` 有没有 `+cpu` |
| 同上 | 容器里没开 GPU | 检查 `--gpus all` / `runtime: nvidia` |
| 同上 | 宿主机驱动没装或太老 | 先在宿主机 `nvidia-smi` 确认 |
| `is_available()` 是 `True` 但一算就崩 | 算力和 PyTorch 版本不匹配 | 看 `get_arch_list()` 里有没有你的 sm，见 X.4.2 |
| `no kernel image is available` | 同上 | 换 PyTorch 版本，别改代码 |
| `CUDA capability sm_xxx is not compatible` | 同上 | 同上 |
| `pip install torch` 装出来的不是想要的版本 | 没加 `--index-url` | 见 X.3.3 |
| `Expected all tensors to be on the same device` | 有张量忘了 `.to(device)` | 见 X.5.2 |
| `nvrtc: failed to open libnvrtc-builtins.so` | pip 装的 NVIDIA 库不在库搜索路径里 | 把 `dist-packages/nvidia/*/lib` 加进 `LD_LIBRARY_PATH` |
| `libcudnn.so.x: cannot open shared object file` | 同上，库路径问题 | 查 `LD_LIBRARY_PATH` |
| `CUDA driver version is insufficient` | 驱动太老 | 装新驱动 |
| `CUDA error: device-side assert triggered` | 通常是索引越界（比如标签数超过类别数） | 检查数据集类别配置 |
| 测出来 20000 FPS | 没同步 | 加 `torch.cuda.synchronize()`，见 X.6 |
| 耗时全堆在 `.cpu()` 那一行 | 那行在替前面的异步操作背锅 | 问题在它前面 |
| 前几帧特别慢 | 冷启动没预热 | 先跑几十次热身 |
| 推理很快但整体 FPS 低 | 瓶颈不在 GPU | 分段计时找瓶颈 |
| `GPU-Util` 一直很低 | GPU 在等数据 | 查相机、预处理、数据加载 |
| 功耗上不去，训练慢 | 节能模式 | `nvidia-settings` 里把 PRIME 切到 Performance，要重启 |
| `CUDA out of memory` | 显存不够 | 减小 batch / 输入尺寸；`nvidia-smi` 看有没有别的进程占着 |

---

## X.8 关键词与命令速查表

### X.8.1 概念名词

| 名词 | 一句话说明 |
| --- | --- |
| **CUDA** | NVIDIA 的 GPU 通用计算平台 |
| **NVIDIA 驱动** | 让系统认识显卡的底层软件，**必须装在宿主机** |
| **CUDA Toolkit** | 开发套件（`nvcc` 等）。**只做推理通常不需要** |
| **CUDA Runtime** | 程序运行时调用的库，pip 版 PyTorch 自带 |
| **cuDNN** | NVIDIA 的深度学习算子库，PyTorch 自带 |
| **TensorRT** | 推理优化引擎，能再快几倍。**生成的 engine 与硬件绑定，不能跨设备** |
| **Kernel（核函数）** | 一段会被几千个 GPU 线程同时执行的代码 |
| **Compute Capability / sm_xx** | 显卡的"指令集方言"版本，如 sm_89、sm_120 |
| **FP32 / FP16 / INT8** | 32/16/8 位数值精度，越低越快越省显存，精度越差 |
| **OOM** | Out of Memory，显存不够 |

### X.8.2 PyTorch CUDA API

| 写法 | 一句话说明 |
| --- | --- |
| `torch.__version__` | 版本字符串，**看 `+cu126` / `+cpu` 后缀** |
| `torch.cuda.is_available()` | 有没有可用 GPU。**返回 True 不代表一定能跑** |
| `torch.cuda.device_count()` | 有几块 GPU |
| `torch.cuda.get_device_name(0)` | 显卡型号 |
| `torch.cuda.get_device_capability()` | 计算能力，`(12, 0)` 即 sm_120 |
| `torch.cuda.get_arch_list()` | **这个 PyTorch 支持哪些算力**，排查版本问题必看 |
| `torch.version.cuda` | PyTorch 用的 CUDA 运行时版本 |
| `torch.backends.cudnn.version()` | cuDNN 版本 |
| `torch.device('cuda')` | 构造设备对象 |
| `x.to(device)` / `x.cpu()` | 搬张量 |
| `x.device` | 查这个张量在哪 |
| `torch.randn(3, 4, device='cuda')` | 直接在 GPU 上创建，省一次拷贝 |
| `x.detach().cpu().numpy()` | 取结果三连，顺序不能乱 |
| `torch.cuda.synchronize()` | **等 GPU 干完活**，计时必备 |
| `model.eval()` | 切推理模式。忘了结果会错 |
| `torch.no_grad()` | 不记录梯度，省显存也快一点 |
| `torch.cuda.memory_allocated()` | 当前占用的显存（字节） |
| `torch.cuda.max_memory_allocated()` | 峰值占用，调 batch 时有用 |
| `torch.cuda.empty_cache()` | 把缓存还给驱动。**救不了自己的 OOM** |

### X.8.3 命令行工具

| 命令 | 一句话说明 |
| --- | --- |
| `nvidia-smi` | 看显卡状态。**右上角 CUDA Version 是驱动支持的上限** |
| `watch -n 0.5 nvidia-smi` | 每 0.5 秒刷新一次 |
| `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` | 只看关心的字段，每秒一行 |
| `ubuntu-drivers devices` | 看系统推荐哪个驱动版本 |
| `sudo ubuntu-drivers autoinstall` | 装推荐驱动 |
| `nvcc -V` | 看 CUDA Toolkit 版本（没装就没这命令） |
| `nvidia-settings` | 图形界面调显卡，PRIME 性能模式在这里切 |

### X.8.4 常见报错对照

| 报错关键词 | 真正的原因 |
| --- | --- |
| `couldn't communicate with the NVIDIA driver` | 驱动没装好，或装完没重启 |
| `no kernel image is available` | 显卡算力不在 PyTorch 支持列表里 |
| `CUDA capability sm_xxx is not compatible` | 同上 |
| `Expected all tensors to be on the same device` | 有张量忘了搬到 GPU |
| `CUDA out of memory` | 显存不够 |
| `CUDA error: device-side assert triggered` | 通常是索引越界 |
| `CUDA driver version is insufficient` | 驱动太老 |
| `libcudnn.so / libnvrtc-builtins.so 打不开` | 库路径问题，查 `LD_LIBRARY_PATH` |

---

## 本章练习

1. 在你的机器上装好驱动，跑通 `nvidia-smi`，记下驱动版本和右上角的 CUDA 版本。
2. 从 PyTorch 官网生成适合你显卡的安装命令，装好之后打印 `torch.__version__`，确认后缀是对的。
3. 跑通 X.3.4 的验证脚本，把输出贴进笔记。
4. 对比 `nvidia-smi` 右上角、`nvcc -V`（如果有）、`torch.version.cuda` 三个数字，解释为什么它们可以不一样。
5. 查出自己显卡的算力，和 `get_arch_list()` 对照，确认版本是匹配的。
6. 故意写一段"CPU 张量 + GPU 张量"的代码，看看报错长什么样，然后修好它。
7. 写一段矩阵乘法（4096×4096），分别在 CPU 和 GPU 上测耗时。**记得加同步。**
8. 把第 7 题里的 `torch.cuda.synchronize()` 去掉，看看测出来的 FPS 有多离谱。
9. 在 6 章那个容器里重跑一遍验证脚本，确认容器里也能用上显卡。

---

## 本章小结

这一章能压成两块。

**装环境：**

```text
   1. 宿主机装驱动        → nvidia-smi 能出信息
   2. 官网生成命令装 PyTorch → --index-url 不能省
   3. 跑验证脚本          → 必须真的算一次，不能只看 is_available()
```

**排查：**

```text
   报错从下往上查：驱动 → 容器权限 → PyTorch 版本 → 代码
   算力对不上就换 PyTorch 版本，别改代码
   测速必须 synchronize + 预热
```

三个最容易犯的错，避开就算这一章到位了：

- **把 `nvidia-smi` 右上角的数字当成"已安装的 CUDA 版本"**，然后照着它去装东西。
- **拿 `is_available()` 当环境正常的证据。** 它只说明 CUDA 初始化成功了。
- **不加同步就测速。** 测出来的 FPS 好看得不真实，然后基于假数据做优化决策。

---

## 参考资料

- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)（装之前先来这里）
- [NVIDIA 驱动下载](https://www.nvidia.cn/drivers/lookup/)
- [CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)
- [CUDA 显卡算力对照表](https://developer.nvidia.com/cuda-gpus)
- [PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)（异步执行、显存管理讲得最清楚）
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（想了解 kernel 和线程模型时看）
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- 硬件层面的 CPU / GPU 差异：本教材 3.0 节
- 容器里怎么把显卡透传进去：本教材 6 章
