# CUDA入门

> 编号占位说明：本章内部编号统一写成 `X.1`、`X.2`……定好章号之后，把全文的 `X.` 全部替换成对应数字即可。

到这一步，你已经能把环境装好、能让程序看见画面了。接下来要面对的问题是：**跑得动，但跑不快。**

一个 YOLO 模型在 CPU 上推理一张图可能要几百毫秒，机器人根本等不起。同样的模型丢到显卡上，可能只要一两毫秒。这一章就是讲这中间发生了什么，以及怎么让它真的发生在你的机器上。

先说清楚这一章**不讲什么**：我们不会教你写 CUDA C++ 的算子，那是另一个方向的事，短期内你也用不上。这一章的目标是让你**看懂那一堆版本号、看懂报错、知道性能卡在哪**。这三件事才是实际开发里天天遇到的。

本章出现的所有名词、命令和 API，在 **X.10 速查表**里都有一句话解释。看正文时遇到不认识的名字，翻到那边对一下就行。

（这里放一张"CPU 一个人搬砖 vs GPU 一群人搬砖"的对比图，开学补图）

## 本章学习目标

完成本章后，你应当能够：

- 解释 GPU 为什么比 CPU 快，以及什么样的任务用 GPU 才划算；
- 分清 **NVIDIA 驱动**、**CUDA Toolkit**、**cuDNN**、**PyTorch 自带的 CUDA 运行时**这四者的关系，不再被版本号搞晕；
- 看懂 `nvidia-smi` 的输出，特别是右上角那个 CUDA 版本号到底是什么意思；
- 说出你的显卡的**计算能力（sm_xx）**，并解释 `no kernel image is available` 这个报错的成因；
- 在 PyTorch 里正确地把模型和数据放上 GPU，并说明为什么 CPU 张量和 GPU 张量不能直接一起算；
- 说明 CUDA 的**异步执行**特性，并写出正确的 GPU 计时代码；
- 找出推理流程里的性能瓶颈，区分"计算慢"和"搬运慢"；
- 用 `no_grad`、`eval`、半精度、预热等手段把推理速度榨出来；
- 看懂显存不够（OOM）时该怎么办；
- 知道 Jetson 平台和桌面显卡有哪些不一样。

代码部分用 **Python + PyTorch**。有一小节会出现 CUDA C++，只是为了让你看清楚"kernel"到底是个什么东西，不要求你写。

---

## X.1 GPU 为什么快

### X.1.1 不是"更快"，是"更多"

一个常见的误解是"GPU 比 CPU 快"。更准确的说法是：**GPU 的单个核心比 CPU 慢得多，但它的核心多得多。**

```text
        CPU                              GPU

   ┌────┐ ┌────┐                ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
   │核心│ │核心│                ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
   └────┘ └────┘                ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
   ┌────┐ ┌────┐                ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
   │核心│ │核心│                ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
   └────┘ └────┘                └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘

   十几个核心                     几千个核心
   每个都很聪明                   每个都比较笨
   适合复杂的、有分支的逻辑        适合简单的、重复几千遍的运算
```

打个比方：CPU 是几个博士生，什么难题都能做，但人少；GPU 是几千个小学生，只会做加减乘除，但人多。

**一道微积分题**交给博士生，几分钟搞定；交给小学生，永远做不出来。
**十万道加法题**交给博士生，算到天亮；交给小学生，一人一道，一分钟收工。

深度学习里的运算恰好是后一种——**矩阵乘法**，本质就是海量的乘加，每一个都独立、都简单。这就是 GPU 在这个领域碾压 CPU 的根本原因。

### X.1.2 什么任务适合 GPU

不是所有事情丢给 GPU 都会变快。判断标准就一条：**能不能拆成几千份互不依赖的小任务。**

| 任务 | 适合 GPU 吗 | 为什么 |
| --- | --- | --- |
| 神经网络推理/训练 | 非常适合 | 全是矩阵乘法 |
| 图像滤波、卷积、缩放 | 适合 | 每个像素独立算 |
| 大规模点云处理 | 适合 | 每个点独立 |
| 排序、查找 | 一般 | 有数据依赖，加速有限 |
| 读文件、解析 JSON | 不适合 | 本来就是串行的 |
| 你上一章写的那个色块识别 | **通常不适合** | 图太小，搬运开销比算的还多 |

最后一行值得展开说：**GPU 有启动成本**。数据要从内存搬到显存，算完还要搬回来。对一张 640×480 的小图做几次形态学操作，这点计算量还不够抵消搬运时间，用 GPU 反而更慢。

> 这是新手最容易犯的错：以为什么都往 GPU 上放就会快。**先测，再优化。** 后面 X.6 会讲怎么正确地测。

### X.1.3 Kernel 是什么（这一节可以跳过）

你会在报错信息里反复看到 "kernel" 这个词，值得花两分钟搞清楚它指什么。

**Kernel（核函数）就是一段"要在 GPU 上被几千个线程同时执行"的代码。**

看一个最小的例子，它把两个数组相加：

```cpp
// 这段是 CUDA C++，看看就好，不要求写
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 我是第几号线程？
    if (i < n) {
        c[i] = a[i] + b[i];                          // 我只负责这一个元素
    }
}
```

关键在于：**这段代码描述的是"一个线程要干的活"**，而不是一个循环。启动时你告诉 GPU"开 10000 个线程跑这段代码"，每个线程通过 `threadIdx` 知道自己是几号、该处理哪个元素。

```text
   CPU 的写法（串行）              GPU 的写法（并行）

   for (i = 0; i < 10000; i++)    开 10000 个线程，每个执行一次
       c[i] = a[i] + b[i];        线程 0 → c[0] = a[0] + b[0]
                                  线程 1 → c[1] = a[1] + b[1]
   一个一个来，10000 步            线程 2 → c[2] = a[2] + b[2]
                                       ...  同时进行
```

好消息是：**我们基本不需要自己写 kernel**。PyTorch、OpenCV、TensorRT 里已经有人写好了成千上万个高度优化的 kernel，我们的工作是把它们串起来用。

知道这个概念的实际价值在于，你以后看到这类报错时不会一脸茫然：

```text
no kernel image is available for execution on the device
```

它说的是："这个包里没有能在你这块显卡上运行的 kernel 二进制代码。" 为什么会没有，X.3 会讲。

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

> 那 Docker 那一章的 Dockerfile 里为什么还装了 `cuda-toolkit-12-6`？因为当时是抱着"以后可能要编译点什么"的想法装的，属于宁可多装。代价是镜像胖了好几个 GB、构建慢了很多。如果你现在从头写一份只做推理的 Dockerfile，这一段完全可以省掉。

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

## X.3 计算能力：sm_xx 是什么

### X.3.1 每一代显卡有一个"方言"

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

### X.3.2 `no kernel image` 到底怎么回事

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

**最坑的地方是：`torch.cuda.is_available()` 返回 `True`。** 驱动认得显卡、CUDA 也初始化成功了，只有真正要执行算子的那一刻才崩。所以不能只靠 `is_available()` 判断环境好没好。

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

> 这就是 Docker 那一章里 `GPU_SERIES` 分支的由来：40 系（sm_89）用一套 whl，50 系（sm_120）用另一套。新硬件刚出来的那段时间，稳定版框架往往还没跟上，只能用预览版。

---

## X.4 在 PyTorch 里用 GPU

### X.4.1 设备的概念

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

### X.4.2 最常见的报错

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

### X.4.3 结果拿回 CPU

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

## X.5 一个容易被忽略的性质：CUDA 是异步的

这一节很关键，不知道的话你测出来的所有性能数据都是错的。

### X.5.1 GPU 调用会"立刻返回"

当你写：

```python
y = model(x)
```

CPU 并不会等 GPU 算完。它只是把任务**扔进一个队列**就返回了，然后继续执行下一行 Python。GPU 在后台慢慢算。

```text
   CPU 时间线：
   ├─ 提交任务1 ─┬─ 提交任务2 ─┬─ 提交任务3 ─┬─ print(耗时) ← 这里只过了 0.1ms！
                │             │             │
   GPU 时间线：  └── 任务1 ────┴── 任务2 ────┴── 任务3 ────────▶ 还在算
```

这个设计是为了效率——CPU 不用干等着，可以继续准备下一批数据。

### X.5.2 错误的计时 vs 正确的计时

**错误写法**（几乎所有人第一次都这么写）：

```python
import time

t0 = time.time()
y = model(x)
print(f"耗时 {(time.time()-t0)*1000:.2f} ms")   # 会打印一个荒谬的小数字
```

你会得到 0.05ms 之类的结果，然后兴高采烈地以为自己有 20000 FPS。实际上你测的只是"把任务扔进队列"的时间。

**正确写法**：加上同步，等 GPU 真的算完：

```python
import time
import torch

torch.cuda.synchronize()          # 先确保之前的活都干完了
t0 = time.time()

y = model(x)

torch.cuda.synchronize()          # 等这次的活干完
print(f"耗时 {(time.time()-t0)*1000:.2f} ms")
```

### X.5.3 有些操作会隐式同步

不是所有时候都要手动 `synchronize`。**任何需要把 GPU 数据读回 CPU 的操作，都会自动等 GPU 算完**：

```python
loss.item()          # 隐式同步
tensor.cpu()         # 隐式同步
tensor.numpy()       # 隐式同步
print(tensor)        # 隐式同步
```

这解释了一个常见的困惑：**"为什么我的耗时都堆在 `.cpu()` 这一行？"**

其实不是 `.cpu()` 慢，是它在替前面所有异步操作背锅——前面那些行提交完就返回了，真正的等待发生在这里。

> 排查性能问题时，看到某一行 `.item()` 或 `.cpu()` 特别慢，**不要去优化那一行**，问题在它前面。

---

## X.6 正确地测性能

### X.6.1 一个能用的测速模板

```python
import time
import torch

def benchmark(model, x, warmup=20, runs=100):
    model.eval()
    with torch.no_grad():
        # 1. 预热
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        # 2. 正式计时
        t0 = time.time()
        for _ in range(runs):
            model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0

    print(f"平均 {dt/runs*1000:.2f} ms  |  {runs/dt:.1f} FPS")
```

### X.6.2 为什么要预热

前几次推理一定比后面慢，有时候慢十倍，原因有好几个：

- **CUDA 上下文初始化**：第一次调用 GPU 时要做一堆准备工作；
- **cuDNN 算法选择**：cuDNN 会为你的输入尺寸挑一个最快的卷积算法，第一次要试；
- **显存分配**：PyTorch 的显存分配器第一次要向驱动申请，之后会缓存复用；
- **JIT / 图编译**：某些模型第一次跑会现场编译。

所以**跑几十次热身之后再计时**，得到的才是稳态性能。反过来说，如果你的程序是"启动后只推理一次"，那就得把冷启动时间也算进去。

> 实际部署时这一点很重要：机器人开机之后，最好先拿一张假图片跑几遍模型预热，或者提前预开启，别等真正要用的时候才第一次调用。

### X.6.3 分段测，找出真正的瓶颈

整体 FPS 低，不代表模型慢。一个典型的视觉流程是这样的：

```text
   读一帧  →  预处理  →  搬到GPU  →  推理  →  搬回CPU  →  后处理  →  显示
   (相机)    (resize)    (H2D)      (GPU)    (D2H)       (NMS等)   (imshow)
```

这七步里任何一步都可能是瓶颈。分开测：

```python
import time, torch

def t():
    torch.cuda.synchronize()
    return time.time()

t0 = t(); frame = cap.read()[1]           ; t1 = t()
inp = preprocess(frame)                    ; t2 = t()
inp = inp.to('cuda')                       ; t3 = t()
with torch.no_grad(): out = model(inp)     ; t4 = t()
out = out.cpu()                            ; t5 = t()
boxes = postprocess(out)                   ; t6 = t()

print(f"读帧 {(t1-t0)*1000:.1f} | 预处理 {(t2-t1)*1000:.1f} | "
      f"上传 {(t3-t2)*1000:.1f} | 推理 {(t4-t3)*1000:.1f} | "
      f"下载 {(t5-t4)*1000:.1f} | 后处理 {(t6-t5)*1000:.1f}")
```

新手常见的意外发现：

| 发现 | 说明 |
| --- | --- |
| 推理只要 2ms，但整体只有 30 FPS | 瓶颈在相机。相机本身就只有 30fps，或者没设 MJPG |
| 预处理比推理还慢 | `cv2.resize` + 归一化在 CPU 上做的，考虑批量或换 GPU 版 |
| 后处理占了一半时间 | NMS 或者画框太慢，检查有没有在循环里做低效操作 |
| `GPU-Util` 只有 20% | GPU 在等数据，瓶颈在 CPU 侧或 IO |

**这是这一章最想让你带走的习惯：先测，再优化。** 凭感觉优化，十次有八次优化错了地方。

---

## X.7 让推理更快

按"性价比"从高到低排。

### X.7.1 关掉梯度（必做，零成本）

推理时不需要反向传播，关掉能省显存也能提速：

```python
model.eval()                    # 切换到推理模式（影响 BN、Dropout）
with torch.no_grad():           # 不记录计算图
    out = model(x)
```

这两句都必须有，作用不一样：

- `model.eval()` 改变**层的行为**：BatchNorm 用running统计量而不是当前批次，Dropout 关闭。**忘了它，推理结果会不对**，而且不会报错，非常隐蔽。
- `torch.no_grad()` 改变**是否记录梯度**：不建计算图，省显存也快一点。

更新一点的写法是 `torch.inference_mode()`，比 `no_grad` 更彻底一些，用法一样。

### X.7.2 半精度 FP16

默认所有计算用 32 位浮点（FP32）。改成 16 位（FP16）通常能快接近一倍，显存也减半：

```python
model = model.half()
x = x.half()
out = model(x)
```

代价是精度下降。对目标检测这类任务，实测可能影响很小——框的位置差个零点几像素，完全不影响使用。**但一定要自己验证一遍**，比较 FP32 和 FP16 的输出差多少再决定。

> 注意 FP16 加速依赖硬件的 Tensor Core。RTX 20 系之后的显卡和 Jetson Orin 都有，很老的卡上开 FP16 可能不快甚至更慢。

训练的话不要手动 `.half()`，用自动混合精度（AMP）：

```python
from torch.amp import autocast, GradScaler
```

这块超出本章范围，先知道有这么个东西。

### X.7.3 batch：实时任务里往往帮不上忙

批量推理（一次送进去 8 张图）能显著提高 GPU 利用率和总吞吐量。**但对实时机器人任务通常没用**，因为：

- 你一次只有一帧图，凑不齐 8 张；
- 硬凑就得等，等待时间比省下来的还多。

所以我们的实时管线基本都是 `batch=1`，GPU 利用率不高是正常的，不用为此焦虑。批量推理主要用在**离线跑数据集**和**训练**的时候。

### X.7.4 减少 CPU 和 GPU 之间的搬运

显存和内存之间的数据传输走 PCIe，比显存内部的访问慢一两个数量级。原则是：

- **数据一次搬上去，尽量在 GPU 上做完所有事**，别搬来搬去；
- **不要在循环里频繁 `.cpu()`**，尤其不要为了打印一个中间值就同步一次；
- 能在 GPU 上做的预处理就在 GPU 上做。

```python
# 不好：每帧都在循环里打印，每次都强制同步
for frame in stream:
    out = model(prep(frame).to(device))
    print(out.max().item())          # ← 每帧同步一次

# 好：只在需要的时候取回
for i, frame in enumerate(stream):
    out = model(prep(frame).to(device))
    if i % 100 == 0:
        print(out.max().item())
```

### X.7.5 再往上就是 TensorRT

如果 PyTorch 层面榨干了还不够快，下一步是 **TensorRT**——NVIDIA 的推理优化引擎。它会做算子融合、精度校准、针对你这块具体显卡挑最优 kernel，通常还能再快两三倍。

典型流程：

```text
   PyTorch 模型  ──export──▶  ONNX  ──trtexec──▶  TensorRT engine
     (.pt)                    (.onnx)              (.engine)
```

两个必须知道的限制：

1. **engine 文件是和硬件绑定的**。在 5080 上生成的 engine，拿到 Jetson 上跑不了，必须在目标设备上重新生成；
2. **生成过程比较慢**，几分钟到几十分钟，所以是提前离线做好、部署时直接加载。

Ultralytics 把这套封装得很简单：

```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine', half=True)     # 在目标设备上执行
```

TensorRT 的细节可以单开一章，这里知道它的定位和"不能跨设备"这两点就够了。

---

## X.8 显存不够怎么办

### X.8.1 认识这个报错

```text
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 16.00 GiB total capacity;
13.50 GiB already allocated; 1.20 GiB free; 14.00 GiB reserved in total by PyTorch)
```

读法：**总共 16G，已经用了 13.5G，还想再要 2G，只剩 1.2G，失败。**

注意 `reserved`（14G）比 `allocated`（13.5G）大——PyTorch 会缓存已经申请过的显存块以便复用，不会立刻还给驱动。所以 `nvidia-smi` 看到的占用通常比你实际用的多，这是正常的。

### X.8.2 显存都被谁占了

```text
   显存 16GB
   ├── 模型权重         ← 固定开销，模型多大就占多大（FP16 减半）
   ├── 激活值           ← 跟 batch size 和输入分辨率成正比，训练时是大头
   ├── 梯度 + 优化器状态 ← 只有训练才有，Adam 大概是权重的 2~3 倍
   └── PyTorch 缓存     ← 可以用 empty_cache() 还回去一部分
```

### X.8.3 按顺序试这些办法

| 办法 | 适用 | 代价 |
| --- | --- | --- |
| 确认用了 `no_grad` | 推理 | 无，本来就该做 |
| 调小 batch size | 训练 | 训练慢一点 |
| 调小输入分辨率 | 都行 | 小目标可能检测不到 |
| 换 FP16 | 都行 | 精度略降 |
| `torch.cuda.empty_cache()` | 都行 | 只能清缓存，治标 |
| 梯度累积 | 训练 | 用小 batch 模拟大 batch，变慢 |
| 换更小的模型（`n` 而不是 `x`） | 都行 | 精度降 |

有一件事要说清楚：

> **`torch.cuda.empty_cache()` 不能解决你自己的 OOM。** 它把 PyTorch 缓存但没在用的显存还给驱动，好处是**别的进程**能用上。你自己再要显存时，PyTorch 本来就会先用缓存池里的。所以指望它救 OOM 基本是徒劳，该减 batch 还是得减。

还有个很实际的检查：**你的显卡上是不是还跑着别的东西？** 桌面环境、浏览器、上一个没退干净的 Python 进程都在吃显存。

```bash
nvidia-smi                      # 看 Processes 那一栏有哪些进程
kill -9 <占着不放的 PID>
```

在 Docker 里跑的时候尤其容易出现"上一个容器还开着"的情况。

---

## X.9 Jetson 有什么不一样

我们的部署目标是 Jetson Orin Nano，它和桌面显卡有几个关键差异，提前知道能少走弯路。

### X.9.1 统一内存

桌面机上，内存和显存是两块物理上分开的东西，数据要通过 PCIe 搬来搬去。**Jetson 上 CPU 和 GPU 共用同一块物理内存**：

```text
      桌面机                              Jetson

  ┌────────┐   PCIe   ┌────────┐      ┌──────────────────┐
  │ 内存   │ ◀──────▶ │ 显存   │      │   统一内存 8GB    │
  │ 64GB   │  慢      │ 16GB   │      │  CPU 和 GPU 共用  │
  └────────┘          └────────┘      └──────────────────┘
```

带来两个后果：

- **好消息**：`.to('cuda')` 的开销小很多，几乎没有真正的"搬运"；
- **坏消息**：**内存和显存互相抢**。8GB 要同时装下操作系统、你的 ROS2 节点、模型权重和中间结果。桌面上不用想的内存问题，在这里会变成天天要想的问题。

这也是 Docker 那一章里 Jetson 服务的 `shm_size` 设成 4GB 而不是 8GB 的原因——共享内存占太多，别的就不够了。

### X.9.2 没有 nvidia-smi

在 Jetson 上敲 `nvidia-smi` 通常会告诉你没这个命令，或者信息残缺。替代品：

```bash
tegrastats                  # 系统自带，输出一行行的实时状态

sudo pip3 install jetson-stats
sudo jtop                   # 图形化，好看好用，强烈推荐
```

`jtop` 能一屏看到 CPU/GPU 占用、内存、温度、功耗模式，是 Jetson 上最常用的工具。

### X.9.3 功耗模式必须手动开

Jetson 出厂默认是省电模式，算力会被限制得很惨。开工前先解放它：

```bash
sudo nvpmodel -q            # 查看当前模式
sudo nvpmodel -m 0          # 切到最高性能模式（0 通常是 MAXN）
sudo jetson_clocks          # 把所有时钟锁到最高频
```

`nvpmodel` 的设置重启后保留，`jetson_clocks` 不保留，需要开机自启的话得写进服务里。

> 桌面机上也有类似的坑。Ubuntu 下的 NVIDIA 驱动默认是 On-Demand 节能模式，我们测过一次训练时功耗卡在 80W 上不去（上限 175W）。在 `nvidia-settings` 里把 PRIME Profiles 切到 NVIDIA (Performance Mode)、重启之后才满血。**性能不对劲的时候，先怀疑是不是根本没让硬件全速跑。**

### X.9.4 架构不同，包不通用

Jetson 是 **ARM64（aarch64）**，桌面机是 **x86_64**。这意味着：

- x86 上下载的 PyTorch whl **在 Jetson 上装不了**，必须用 NVIDIA 为 JetPack 编译的专用版本；
- x86 上构建的 Docker 镜像在 Jetson 上跑不起来，要用 `l4t` 系列的基础镜像单独构建；
- TensorRT engine 更是完全不能跨过去，必须在 Jetson 上重新生成。

计算能力也不一样：Jetson Orin 是 **sm_87**。

> 一句话总结：**Jetson 是另一个平台，不是"性能弱一点的电脑"。** 环境要单独准备一套，测试也要在真机上做。这也是我们 compose 文件里单独留了一个 jetson 服务的原因。

---

## X.10 常见问题排查表

| 现象 | 可能原因 | 怎么办 |
| --- | --- | --- |
| `torch.cuda.is_available()` 是 `False` | 装的是 CPU 版 PyTorch | `print(torch.__version__)`，带 `+cpu` 就是装错了 |
| 同上 | 容器里没开 GPU | 检查 `--gpus all` / `runtime: nvidia` |
| 同上 | 宿主机驱动没装或太老 | 先在宿主机 `nvidia-smi` 确认 |
| `is_available()` 是 `True` 但一算就崩 | 算力和 PyTorch 版本不匹配 | 看 `get_arch_list()` 里有没有你的 sm，见 X.3.2 |
| `no kernel image is available` | 同上 | 换 PyTorch 版本，别改代码 |
| `Expected all tensors to be on the same device` | 有张量忘了 `.to(device)` | 见 X.4.2 |
| `nvrtc: failed to open libnvrtc-builtins.so` | pip 装的 NVIDIA 库不在库搜索路径里 | 把 `dist-packages/nvidia/*/lib` 加进 `LD_LIBRARY_PATH` |
| 测出来 20000 FPS | 没同步，测的是提交任务的时间 | 加 `torch.cuda.synchronize()`，见 X.5.2 |
| 耗时全堆在 `.cpu()` 那一行 | 那行在替前面的异步操作背锅 | 问题在它前面，见 X.5.3 |
| 前几帧特别慢 | 冷启动，还没预热 | 先跑几十次热身，见 X.6.2 |
| 推理很快但整体 FPS 低 | 瓶颈不在 GPU | 分段计时，见 X.6.3 |
| `GPU-Util` 一直很低 | GPU 在等数据 | 查相机、预处理、数据加载 |
| 推理结果不对但不报错 | 忘了 `model.eval()` | 见 X.7.1 |
| `CUDA out of memory` | 显存不够 | 见 X.8.3；先确认没有别的进程占着 |
| 显存明明够却 OOM | 碎片化，或别的进程占着 | `nvidia-smi` 看 Processes；重启进程 |
| 功耗上不去，训练慢 | 节能模式 | 桌面：`nvidia-settings` PRIME；Jetson：`nvpmodel -m 0` |
| Jetson 上 `nvidia-smi` 用不了 | 平台不支持 | 用 `tegrastats` 或 `jtop`，见 X.9.2 |
| x86 上下的 whl 在 Jetson 装不上 | 架构不同 | 用 JetPack 版 PyTorch，见 X.9.4 |
| TensorRT engine 换台机器就报错 | engine 和硬件绑定 | 在目标设备上重新导出 |

---

## X.11 关键词与 API 速查表

### X.11.1 概念名词

| 名词 | 一句话说明 |
| --- | --- |
| **CUDA** | NVIDIA 的 GPU 通用计算平台和编程模型 |
| **NVIDIA 驱动** | 让系统认识显卡的底层软件，**必须装在宿主机** |
| **CUDA Toolkit** | 开发套件（`nvcc` 编译器、头文件、工具）。**只做推理通常不需要** |
| **CUDA Runtime** | 程序运行时调用的那套库，pip 版 PyTorch 自带 |
| **cuDNN** | NVIDIA 的深度学习算子库，PyTorch 自带 |
| **TensorRT** | 推理优化引擎，能再快几倍，**engine 与硬件绑定** |
| **Kernel（核函数）** | 一段会被几千个 GPU 线程同时执行的代码 |
| **Compute Capability / sm_xx** | 显卡的"指令集方言"版本，如 sm_89、sm_120 |
| **FP32 / FP16 / INT8** | 32/16/8 位数值精度，越低越快越省显存，精度越差 |
| **Tensor Core** | 专门做混合精度矩阵运算的硬件单元，FP16 加速靠它 |
| **H2D / D2H** | Host to Device / Device to Host，内存↔显存的数据搬运 |
| **OOM** | Out of Memory，显存不够 |
| **统一内存** | Jetson 上 CPU 和 GPU 共用同一块物理内存 |
| **JetPack** | NVIDIA 给 Jetson 的系统与软件套件 |
| **L4T** | Linux for Tegra，Jetson 的系统基础 |

### X.11.2 PyTorch CUDA API

| 写法 | 一句话说明 |
| --- | --- |
| `torch.cuda.is_available()` | 有没有可用的 GPU。**返回 True 不代表一定能跑** |
| `torch.cuda.device_count()` | 有几块 GPU |
| `torch.cuda.get_device_name(0)` | 显卡型号 |
| `torch.cuda.get_device_capability()` | 计算能力，如 `(12, 0)` 表示 sm_120 |
| `torch.cuda.get_arch_list()` | **这个 PyTorch 支持哪些架构**，排查版本问题必看 |
| `torch.version.cuda` | PyTorch 用的 CUDA 运行时版本 |
| `torch.backends.cudnn.version()` | cuDNN 版本 |
| `torch.device('cuda')` | 构造设备对象 |
| `x.to(device)` / `x.cuda()` / `x.cpu()` | 把张量搬到指定设备 |
| `x.device` | 查这个张量在哪 |
| `torch.randn(3, 4, device='cuda')` | **直接在 GPU 上创建**，比先建后搬省一次拷贝 |
| `x.detach().cpu().numpy()` | 取结果的标准三连，顺序不能乱 |
| `torch.cuda.synchronize()` | **等 GPU 把活干完**，计时必备 |
| `model.eval()` | 切推理模式，影响 BN/Dropout。**忘了结果会错** |
| `torch.no_grad()` | 不记录梯度，省显存提速 |
| `torch.inference_mode()` | 比 `no_grad` 更彻底的推理模式 |
| `model.half()` / `x.half()` | 转 FP16 |
| `torch.cuda.memory_allocated()` | 当前实际占用的显存（字节） |
| `torch.cuda.memory_reserved()` | PyTorch 向驱动要了多少（含缓存） |
| `torch.cuda.empty_cache()` | 把缓存还给驱动。**救不了自己的 OOM** |
| `torch.cuda.max_memory_allocated()` | 峰值占用，调 batch size 时很有用 |

### X.11.3 命令行工具

| 命令 | 一句话说明 |
| --- | --- |
| `nvidia-smi` | 看显卡状态。**右上角 CUDA Version 是驱动支持的上限** |
| `watch -n 0.5 nvidia-smi` | 每 0.5 秒刷新一次 |
| `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` | 只看关心的字段，每秒一行 |
| `nvcc -V` | 看 CUDA Toolkit 版本（没装 Toolkit 就没这个命令） |
| `nvidia-settings` | 图形界面调显卡，PRIME 性能模式在这里切 |
| `tegrastats` | **Jetson** 上的实时状态输出 |
| `jtop` | **Jetson** 上的图形化监控，需 `pip3 install jetson-stats` |
| `sudo nvpmodel -m 0` | **Jetson** 切到最高性能模式 |
| `sudo jetson_clocks` | **Jetson** 锁定最高时钟频率（重启失效） |
| `trtexec --onnx=x.onnx --saveEngine=x.engine` | 把 ONNX 转成 TensorRT engine |

### X.11.4 常见报错对照

| 报错关键词 | 真正的原因 |
| --- | --- |
| `no kernel image is available` | 显卡算力不在 PyTorch 支持列表里 |
| `CUDA capability sm_xxx is not compatible` | 同上，更直白的版本 |
| `Expected all tensors to be on the same device` | 有张量忘了搬到 GPU |
| `CUDA out of memory` | 显存不够 |
| `CUDA error: device-side assert triggered` | 通常是索引越界（比如标签数超过类别数） |
| `CUDA driver version is insufficient` | 驱动太老，装新驱动 |
| `libcudnn.so.x: cannot open shared object file` | 库路径问题，查 `LD_LIBRARY_PATH` |
| `nvrtc: failed to open libnvrtc-builtins.so` | 同上，pip 装的 NVIDIA 库没在搜索路径里 |

---

## 本章练习

1. 在你的机器上跑一遍 X.3.2 的自检脚本，记下显卡型号、算力、`get_arch_list()` 的内容。
2. 对比 `nvidia-smi` 右上角的 CUDA 版本、`nvcc -V`（如果有）、`torch.version.cuda` 三个数字，解释为什么它们可以不一样。
3. 写一段矩阵乘法（比如 4096×4096），分别在 CPU 和 GPU 上跑，比较耗时。**记得加同步。**
4. 把第 3 题的矩阵尺寸从 64 开始逐步加大到 8192，画一条"尺寸 vs GPU 相对 CPU 的加速比"曲线。找出加速比开始大于 1 的那个尺寸，并解释为什么小矩阵反而是 CPU 快。
5. 故意把 X.5.2 里错误的计时代码跑一遍，看看它报出来的 FPS 有多离谱。
6. 用 X.6.1 的模板测一个 YOLO 模型的推理耗时，分别测 warmup=0 和 warmup=20，比较结果。
7. 用 X.6.3 的分段计时，测一遍你自己的"读帧→预处理→推理→后处理"完整流程，找出瓶颈在哪一段。
8. 把模型转成 FP16 再测一次速度，同时比较 FP32 和 FP16 的检测框坐标差了多少。
9. 故意把 batch size 或输入分辨率调大直到 OOM，读懂报错里的四个数字分别是什么。
10. 进阶：把一个 YOLO 模型导出成 TensorRT engine，测速并与 PyTorch 原生推理对比。（提示：导出要在目标设备上做）

---

## 本章小结

这一章讲的大部分内容，最后能压缩成几条经验：

```text
   版本这件事：
     驱动 ≥ CUDA 运行时，驱动装新的
     nvidia-smi 的 CUDA Version 是"上限"，不是"已装"
     pip 版 PyTorch 自带运行时，通常不用装 Toolkit
     算力对不上就换 PyTorch 版本，别改代码

   性能这件事：
     CUDA 是异步的 → 计时必须 synchronize
     先预热再测 → 前几次不算数
     分段测 → 瓶颈往往不在你以为的地方
     先测，再优化
```

三个最容易犯的错，如果你现在能避开，这一章就到位了：

- **拿 `is_available()` 当环境正常的证据**。它只说明 CUDA 初始化成功了，真正的验证是做一次矩阵乘法。
- **不加同步就测速**。测出来的 FPS 好看得不真实，然后基于这个假数据去做优化决策。
- **推理时忘了 `model.eval()`**。它不报错，只是安静地给你错的结果。

还有一条更大的：**GPU 不是万能加速器。** 它擅长的是"同一件简单的事重复几千遍"，把不适合的任务塞给它只会更慢。判断一件事该不该上 GPU，最靠谱的办法永远是老老实实测一遍。

---

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（想了解 kernel 和线程模型时看）
- [CUDA GPU Compute Capability 对照表](https://developer.nvidia.com/cuda-gpus)
- [PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)（异步执行、显存管理讲得最清楚）
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)（按显卡和 CUDA 版本选对应命令）
- [PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [jetson-stats (jtop)](https://github.com/rbonghi/jetson_stats)
- [Ultralytics 模型导出文档](https://docs.ultralytics.com/modes/export/)
