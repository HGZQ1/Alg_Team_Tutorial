# PyTorch入门

> 编号占位说明：本章内部编号统一写成 `Y.1`、`Y.2`……定好章号之后，把全文的 `Y.` 全部替换成对应数字即可。

## 这一章是干什么的

**PyTorch 是我们跑深度学习模型用的框架。** 你训练出来的每个 `.pt` 文件是它的格式，你写的每行 Ultralytics 代码底下都是它在干活。

**Ultralytics 是 PyTorch 上面的一层封装**，把 YOLO 的训练和推理包成了几行代码。我们平时直接打交道的是它。

这一章要做的事很具体：

```text
   1. 把 PyTorch 装对
   2. 把 Ultralytics 装上，下载一个基础模型，跑通
   3. 知道 YOLO 该选哪个版本、哪个尺寸
   4. 同一份代码怎么在训练机（GPU）和小电脑（CPU）上都能跑
```

**这一章不讲怎么训练 YOLO。** 数据集怎么做、超参怎么调、曲线怎么看、模型怎么评估，这些在有监督学习那一章讲。我们负责的是它下面那一层：

```text
   数据集、标注、超参、训练、评估      ← 有监督学习那一章
   ────────────────────────────────
   环境怎么装、模型怎么选、             ← 我们这一章
   基础模型怎么下、CPU/GPU 怎么切
```

一句话分工：**那一章讲怎么练出模型，这一章讲怎么把环境搭好、把模型跑起来。**

## 本章学习目标

完成本章后，你应当能够：

- 说清 PyTorch、Ultralytics、你自己的节点各负责什么；
- 按机器选对 PyTorch 版本，并看懂 `2.x.x+cu126` 这种版本字符串；
- 装好 Ultralytics，下载基础模型，用命令行和 Python 两种方式跑通一次推理；
- 从任务类型、模型尺寸、模型版本三个维度选出适合当前项目的 YOLO，并说明**为什么不是越新越好**；
- 写出一份同时能在训练机和小电脑上跑的设备配置，不硬编码 `cuda`；
- 说明在 CPU 上跑推理时线程数为什么要管，以及为什么 `.half()` 在 CPU 上反而更慢；
- 打开一个 `.pt` 文件看它是怎么练出来的；
- 说出 ROCm 是什么，以及我们的 AMD 小电脑为什么直接用 CPU。

---

## Y.1 三层结构：谁负责什么

新生最容易懵的一点是：明明只想跑个 YOLO，怎么冒出来这么多东西。把层级摆清楚就好理解了。

```text
   ┌────────────────────────────────────────────┐
   │  你的 ROS2 节点  detector_node.py           │  ← 你写的
   ├────────────────────────────────────────────┤
   │  Ultralytics (YOLO)                        │  ← 把训练/推理封装成几行
   ├────────────────────────────────────────────┤
   │  PyTorch                                   │  ← 张量、模型、GPU 调度
   ├────────────────────────────────────────────┤
   │  CUDA Runtime + cuDNN   /   CPU 算子库      │  ← 真正算数的地方
   ├────────────────────────────────────────────┤
   │  NVIDIA 驱动（走 GPU 才需要）  /  CPU        │
   └────────────────────────────────────────────┘
```

三条由这张图直接推出来的结论：

- **Ultralytics 不是独立框架**，它建立在 PyTorch 上。PyTorch 装错了，Ultralytics 一定跟着出问题；
- **中间那层可以换。** 同一份代码，底下接 CUDA 就走显卡，接 CPU 算子库就走 CPU。这就是我们能在训练机上用 GPU、在小电脑上用 CPU 的原因；
- **报错从下往上查。** `import torch` 就崩，别去看 Ultralytics；`is_available()` 是 False，先去看驱动。

---

## Y.2 装 PyTorch

CUDA 那一章已经讲过完整流程（驱动 → PyTorch → 验证），这里只强调最容易错的两点。

**第一，安装命令去官网生成，别抄旧教程。**

> **PyTorch 安装选择器**：<https://pytorch.org/get-started/locally/>

```bash
# NVIDIA 显卡（CUDA 12.6）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 没有 NVIDIA 显卡（我们的 AMD 小电脑就用这个）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**`--index-url` 不能省。** 不加它，pip 装的多半不是你要的变体。

**第二，装完先看版本字符串。**

```python
import torch
print(torch.__version__)
```

```text
2.5.1+cu126      ← 配 CUDA 12.6
2.5.1+cpu        ← CPU 版，没有 GPU 支持
2.5.1+rocm6.2    ← AMD 的 ROCm 版
```

**加号后面那截决定一切。** 看到 `+cpu` 却在找为什么用不了显卡，那不是环境坏了，是包装错了。这个错每年都有人犯。

详细的验证脚本在 CUDA 那一章 X.3.4，装完照着跑一遍就行。

---

## Y.3 装 Ultralytics，跑通第一个模型

### Y.3.1 安装

```bash
pip3 install ultralytics
```

Ultralytics 会自动把 PyTorch 一起带上。**但如果你要用 GPU，还是建议先按 Y.2 手动装好对应版本的 PyTorch，再装 Ultralytics**——让它自己解析，有可能给你拉一个 CPU 版。

验证：

```bash
yolo checks
```

这条命令会打印出 Ultralytics 版本、PyTorch 版本、CUDA 状态、显卡信息，一次性把环境状况报全。**装完先跑这个。**

> 我们项目里 `requirements.txt` 锁的是 `ultralytics==8.3.0`，而且注释里写了 torch 不显式锁定、交给依赖链按平台解析。这个做法是为了让同一份 requirements 在 NVIDIA 训练机和 AMD 小电脑上都能用，代价是版本不确定。真要复现环境，配合 6 章讲的离线 whl 方案把包固定下来。

### Y.3.2 下载基础模型

**基础模型（预训练权重）是官方在 COCO 数据集上练好的通用模型**，认识 80 类日常物体（人、车、瓶子、椅子……）。它有两个用途：

1. **拿来验证环境**——能跑通就说明整条链路是好的；
2. **当训练的起点**——从预训练权重开始练自己的数据，比从零开始快得多、效果也好得多。

最简单的下载方式是**不用管**，第一次运行会自动下：

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')      # 本地没有就自动下载
```

要手动下（比如网太差）：

> **Ultralytics 权重下载页**：<https://github.com/ultralytics/assets/releases>

下好的 `.pt` 放在脚本同目录，或者用绝对路径指定即可。

**注意文件名的规律**，看名字就知道是什么：

```text
   yolov8    n     .pt
   ───┬──  ──┬──
      │       └── 尺寸：n / s / m / l / x
      └────────── 版本：yolov8 / yolo11 / yolo26 ...

   yolov8n-seg.pt    ← 带 -seg 是实例分割
   yolov8n-pose.pt   ← 带 -pose 是姿态估计
   yolov8n-obb.pt    ← 带 -obb 是旋转框
   yolov8n.pt        ← 什么都不带就是普通目标检测
```

### Y.3.3 跑通：命令行版

最快的验证方式，一条命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' show=True
```

或者用本地图片：

```bash
yolo predict model=yolov8n.pt source=test.jpg save=True
```

结果会存在 `runs/detect/predict/` 下面。能看到画着框的图，**整条链路就通了**。

`yolo` 命令的语法很规整，记住这个结构就会用：

```text
   yolo  [任务]  模式  参数=值 ...

   任务：detect / segment / pose / obb / classify（可省略，会自动判断）
   模式：predict（推理） / train（训练） / val（验证） / export（导出）
```

比如：

```bash
yolo detect predict model=best.pt source=0            # source=0 是摄像头
yolo detect predict model=best.pt source=video.mp4    # 视频文件
yolo export model=best.pt format=onnx                 # 导出成 ONNX
```

### Y.3.4 跑通：Python 版

实际写节点用的是这个：

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model('test.jpg', conf=0.5, verbose=False)

for box in results[0].boxes:
    xyxy = box.xyxy[0].tolist()      # [x1, y1, x2, y2]
    conf = float(box.conf[0])        # 置信度
    cls  = int(box.cls[0])           # 类别编号
    print(f"{model.names[cls]:12s} conf={conf:.2f} box={[round(v) for v in xyxy]}")
```

几个要点：

| 写法 | 说明 |
| --- | --- |
| `model('test.jpg')` | 输入可以是路径、OpenCV 读的 numpy 数组、视频路径、摄像头编号 |
| `conf=0.5` | 置信度阈值，低于这个值的框不要 |
| `verbose=False` | 关掉每帧打印一行的行为。**实时循环里必须加**，不然日志刷屏 |
| `results[0]` | 第一张图的结果 |
| `.boxes` | 所有检测框，含 `.xyxy` / `.conf` / `.cls` |
| `model.names` | 类别编号 → 名字的字典 |
| `results[0].plot()` | 返回一张画好框的图，可以直接 `cv2.imshow` |

配合 OpenCV 接摄像头：

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    cv2.imshow('yolo', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**注意：OpenCV 读出来的 BGR 数组直接传就行，Ultralytics 内部会自己做缩放、通道转换、归一化、搬设备**，不需要你手动处理。

到这一步，环境就算搭完了。

---

## Y.4 YOLO 模型怎么选

这一节是本章除了装环境之外最重要的部分。新生的默认想法是"选最新的"，这个想法**在工程里经常是错的**。

选型要看三个维度：**任务类型 → 模型尺寸 → 模型版本**。按这个顺序定。

### Y.4.1 第一步：任务类型

先想清楚你要的是什么，选错了后面全白费：

| 任务 | 输出 | 什么时候用 | 权重后缀 |
| --- | --- | --- | --- |
| **检测 detect** | 矩形框 + 类别 | 大部分场景，我们用得最多 | 无后缀 |
| **分割 segment** | 像素级轮廓 | 需要知道物体准确形状、要抓取 | `-seg` |
| **姿态 pose** | 关键点 | 人体动作、机械臂关节 | `-pose` |
| **旋转框 obb** | 带角度的框 | 目标斜着放、需要知道朝向 | `-obb` |
| **分类 classify** | 整张图一个标签 | 图里只有一个东西，只要知道是什么 | `-cls` |

**能用检测解决的就用检测。** 分割和姿态的标注成本高好几倍，推理也更慢。我们今年的武器识别和 KFS 识别都是纯检测。

> 如果只需要"这个斜着的方块朝哪边"，先想想能不能用 5 章讲的 `cv2.minAreaRect` 从检测框里算出来——传统方法能解决的，不一定要动模型。

### Y.4.2 第二步：模型尺寸

同一个版本有五个尺寸，从小到大：

```text
   n (nano)  →  s (small)  →  m (medium)  →  l (large)  →  x (extra)

   小 ←─────────── 参数量、精度、显存、耗时 ──────────→ 大
   快 ←─────────────────── 推理速度 ─────────────────→ 慢
```

选法很实在：

| 情况 | 选什么 |
| --- | --- |
| 边缘设备、CPU 推理、要高帧率 | **n** |
| 有独显、要兼顾精度和速度 | **s** 或 **m** |
| 离线跑数据、精度优先 | **l** 或 **x** |
| 不知道选什么 | **从 n 开始**，不够再往上 |

**建议一律从 `n` 起步。** 先把流程跑通、把数据集调好，再考虑换大模型。很多时候你以为需要更大的模型，实际上是数据集有问题——换成 `x` 也救不了。

我们今年两个模型的选择可以当参考：武器识别 3 个类别、目标不小，用了 `yolov8n`；KFS 识别 30 个类别、类间相似度高，用了 `yolov8s`。

### Y.4.3 第三步：模型版本——为什么不是越新越好

Ultralytics 一个包里就支持一大堆版本：YOLOv3、v5、v6、v8、v9、v10、YOLO11、YOLO12、YOLO26，还有 RT-DETR、YOLO-World、YOLOE、SAM 系列。看着就头大。

**先说结论：我们今年用的是 YOLOv8。**

为什么不用最新的？因为选版本要权衡的**不只是精度指标**：

| 要考虑的 | 说明 |
| --- | --- |
| **资料量** | 遇到问题能不能搜到答案。老版本踩过的坑早被人写成博客了，新版本你可能是第一个撞上的 |
| **导出链路成熟度** | 能不能顺利导出成 ONNX / TensorRT。新版本刚出来时，导出经常有各种小问题 |
| **API 稳定性** | 不同版本的参数名和调用方式**可能不一样**，换版本要改代码 |
| **团队现状** | 已有的权重、标注格式、训练脚本、部署代码都是围绕某个版本建起来的 |
| **硬件适配** | 有些新架构在特定硬件上反而更慢 |

YOLOv8 的优势就在这几条上：**它是目前生态最成熟、资料最多、导出链路最稳的一代**，做工程项目很省心。比赛前一天出问题，你需要的是"这个报错网上有一堆人问过"，而不是"这个模型在论文里 mAP 高 0.5"。

各版本的定位大致是这样（细节看官方文档，版本更新很快）：

| 版本 | 定位 |
| --- | --- |
| YOLOv5 | 老资格，资料极多，很多老项目还在用 |
| **YOLOv8** | **生态最成熟，工程首选**，我们今年用的这个 |
| YOLO11 | v8 的稳妥升级，各任务权重齐全，成熟度接近 v8 |
| YOLO12 | 引入注意力机制，偏研究方向 |
| YOLO26 | 2026 年 1 月发布的最新一代，端到端免 NMS，**CPU 推理明显更快** |
| RT-DETR | Transformer 架构的检测器，思路不同 |
| YOLO-World / YOLOE | 开放词汇检测，不用训练就能检测新类别 |
| SAM 系列 | 万物分割，通常配合别的模型用 |

### Y.4.4 那什么时候该换版本

不是说永远别换。给几个值得考虑的信号：

- **CPU 部署帧率不够。** YOLO26 主打的就是边缘设备和 CPU 推理，官方说 CPU 上比前代快不少，而且免 NMS 少一步后处理。我们小电脑那边如果哪天帧率真的顶不住，这是第一个该试的方向；
- **任务变了。** 需要分割、姿态、开放词汇这些 v8 支持不好或者不支持的能力；
- **换了新赛季，可以从头来，也能找到强队成熟方案。** 没有历史包袱的时候，直接上更新的版本是合理的。

但换版本要当成一件正式的事来做，不是改个文件名。**至少要重新过一遍这四步**：

```text
   1. 装对应版本的 ultralytics（版本要求可能不同）
   2. 重新训练（旧版本的权重不能直接给新版本用）
   3. 重新验证导出链路（ONNX / TensorRT 能不能出）
   4. 检查推理代码的参数名和返回结构有没有变
```

第 4 步最容易被忽略。**不同版本的 API 和参数不完全一样**——比如免 NMS 的模型不需要 NMS 相关参数，返回结构也可能有差别。照着 v8 的代码直接跑新模型，很可能不报错但结果不对。

> 顺便提一句许可证：Ultralytics 是 **AGPL-3.0**。校内比赛和学习用完全没问题，但如果哪天涉及闭源商用，得先搞清楚授权。

---

## Y.5 换机器：一份代码，两台机器

我们的实际情况是：**训练机是 NVIDIA，跑 GPU；机器人小电脑是 AMD 7840H，跑 CPU。** 这一节讲怎么让同一份代码在两边都能用。

### Y.5.1 别硬编码 `cuda`

看一眼我们现在的 `detector_params.yaml`：

```yaml
vision_detector:
  ros__parameters:
    model_path: "yolov8n.pt"
    device: "cuda"          # ← 硬编码了
```

这份配置拿到小电脑上跑不起来，得手动改成 `cpu`。改一次不麻烦，但两台机器各存一份不同的配置，迟早会有人拉错分支、改漏一处。

加一个 `auto` 就省心了：

```python
import torch

def pick_device(pref: str = 'auto') -> str:
    """pref: 'auto' / 'cuda' / 'cpu' / '0'"""
    if pref == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda' if pref in ('auto', 'cuda') else pref
    if pref != 'auto':
        print(f"[warn] 请求了 {pref}，但没有可用的 GPU，退回 CPU")
    return 'cpu'
```

配置里写 `device: "auto"`，两台机器都能直接用。

> 那句 `[warn]` 是有意加的：**静默退回 CPU 很危险。** 你以为在用 GPU、实际在用 CPU，然后发现帧率只有十几还找不到原因。宁可吵一点。

### Y.5.2 CPU 上跑推理，要管线程数

这是从 GPU 换到 CPU 之后最需要注意的事。

PyTorch 在 CPU 上默认**用满所有核心**。单跑一个程序没问题，但小电脑上同时跑着一堆 ROS2 节点：视觉、导航、串口、决策……如果每个节点都想占满所有线程，它们会互相抢，结果是**每个都变慢，总吞吐反而下降**。

```text
   不限制：  视觉 ████████████  导航 ████████████  其他 ████████████
                        ↓
             操作系统疯狂切换上下文，谁都跑不快

   限制后：  视觉 ██████        导航 ████        其他 ██
                        ↓
             各干各的，总体更快也更稳
```

两处都要设，少一处都不行：

```python
import torch
import cv2

torch.set_num_threads(4)        # PyTorch 的线程数
cv2.setNumThreads(2)            # OpenCV 也会自己开线程，别忘了它
```

以及环境变量（要在 `import torch` **之前**设，或者写进启动脚本）：

```bash
export OMP_NUM_THREADS=4
```

设多少要测。经验起点：**总核数除以要跑的重节点数**，然后上下调着试。

### Y.5.3 CPU 上不要用 `.half()`

GPU 上开半精度通常能快接近一倍。**但在 CPU 上，`.half()` 大概率让你更慢。**

原因是绝大多数 CPU 没有原生 FP16 计算单元，PyTorch 的 CPU 算子对 FP16 支持也不完整。实际发生的事情是：遇到不支持的算子就悄悄转回 FP32 算完再转回来，**白白多了两次类型转换**。

CPU 上想提速，方向是别的：

| 手段 | 效果 | 代价 |
| --- | --- | --- |
| 降输入尺寸（640 → 480 → 320） | **最直接有效** | 小目标可能丢 |
| 换更小的模型（s → n） | 直接有效 | 精度降 |
| 跳帧（每 2~3 帧推理一次） | 直接有效 | 有延迟 |
| 限制线程数避免抢核 | 见 Y.5.2 | 无 |
| 导出 ONNX Runtime + INT8 量化 | 通常还能再快一截 | 多一套流程 |
| 换 YOLO26 这类专门优化 CPU 的版本 | 见 Y.4.4 | 要重新训练 |

前两项我们**已经在做**，后面几项目前**没做**——小电脑上就是直接加载 `.pt` 跑。这是有意识的取舍：少一套导出流程，就少一个赛前会出问题的环节。

> `requirements.txt` 里其实已经装了 `onnx` 和 `onnxruntime`，路铺好了，只是还没走。等帧率真的不够了再上。

### Y.5.4 期望值管理

从 GPU 换到 CPU，性能会掉一到两个数量级，**这是正常的，不是你配置错了**。

具体多少要自己测。CUDA 那章 X.6 的测速模板在 CPU 上一样能用，只是不需要 `synchronize()`（CPU 计算本来就是同步的）。测的时候记着两件事：

- **先预热几十次再计时**，冷启动那几次不算数；
- **在真实负载下测**，也就是别的 ROS2 节点都开着的时候测。空载数据在赛场上不作数。

---

## Y.6 `.pt` 文件里有什么

我们的小电脑就是直接加载 `.pt` 跑推理，所以得知道这个文件是什么。

**`.pt` 本质是个 ZIP 包**，你可以直接 `unzip -l best.pt` 看里面的条目。Ultralytics 存进去的东西相当全：模型权重、类别名、训练时用的全部超参、逐 epoch 的曲线、训练日期、当时的 ultralytics 版本。

**也就是说，一个权重文件里藏着这个模型的完整身世。** 几行代码就能打开看：

```python
from ultralytics import YOLO

m = YOLO('best.pt')
print("类别   :", m.names)
print("任务   :", m.task)

ck = m.ckpt
print("训练日期:", ck.get('date'))
print("版本   :", ck.get('version'))

args = ck.get('train_args', {})
for k in ('model', 'epochs', 'imgsz', 'batch', 'optimizer', 'device'):
    print(f"  {k:10s}: {args.get(k)}")
```

拿我们仓库 `vision_detector/weights/` 下的权重跑一下，能看到：

```text
best.pt   3 类（W_punch / W_palm / W_spear），基于 yolov8n，imgsz 640
kfs.pt    30 类（FAKE_1~15 + REAL_1~15），基于 yolov8s，imgsz 704
```

**这个习惯建议养成：接手别人的模型，第一件事是打开看看它是怎么练的。** 十秒钟就知道它认识哪些类、输入尺寸多少、什么时候练的，比去群里问一圈快得多。

三件加载时要注意的事：

**一、部署用 `best.pt`，不是 `last.pt`。** 训练会存两个：`last.pt` 是最后一轮，用来断点续训；`best.pt` 是验证集上表现最好的那一轮。训练跑完随手拿了 `last.pt` 就走，这事每年都会发生一次。

**二、`torch.load` 的默认行为变过。** PyTorch 2.6 之后默认只允许加载纯权重，自己写脚本去 `torch.load` 一个完整 checkpoint 会报一段看起来像安全警告的东西：

```text
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel
```

解法是显式关掉：

```python
ck = torch.load('best.pt', weights_only=False)
```

走 `YOLO('best.pt')` 一般不会遇到，自己写脚本才会撞上。**只对来源明确的权重这么干**，从网上随便下的东西别这样加载。

**三、ultralytics 版本要对得上。** 我们 `requirements.txt` 锁的是 `8.3.0`，而仓库里两个权重分别是 `8.3.240` 和 `8.4.14` 存的——**都比锁定版本新**。跨版本加载不一定出问题，但确实可能。自己验一下：

```python
import ultralytics
from ultralytics import YOLO
print("当前:", ultralytics.__version__)
print("权重:", YOLO('best.pt').ckpt.get('version'))
```

差得远的话，要么升 ultralytics，要么用当前版本重新导出一次权重。

---

## Y.7 ROCm 是什么，以及我们为什么不用

我们的小电脑是 **AMD Ryzen 7 7840H**，核显是 Radeon 780M。既然是 AMD，理论上对应 CUDA 的方案叫 **ROCm**。

一张对照表就能建立起概念：

| CUDA 世界 | ROCm 世界 |
| --- | --- |
| `nvidia-smi` | `rocm-smi` |
| `nvcc` | `hipcc` |
| CUDA C++ | HIP |
| `sm_89` / `sm_120` | `gfx1103` / `gfx1151` |
| cuDNN | MIOpen |
| TensorRT | MIGraphX |
| `torch.version.cuda` | `torch.version.hip` |
| `device='cuda'` | **还是 `device='cuda'`** |

最后一行很反直觉，值得单独说：

> **PyTorch 的 ROCm 版本仍然使用 `device='cuda'` 和 `torch.cuda.*` 接口。** HIP 在底下伪装成了 CUDA，上层代码原封不动就能跑。判断自己在哪个后端，看 `torch.version.hip` 是不是 `None`。
>
> 意思是：如果哪天真要迁到 ROCm，**代码基本不用改，工作量全在装环境上。**

**但我们没走这条路，三个原因：**

1. **型号不在支持列表。** ROCm 的硬件支持范围比 CUDA 窄得多。7840H 的核显是 `gfx1103`，而官方兼容矩阵里列的 Ryzen APU 是 `gfx1150`（Ryzen AI 300 系）和 `gfx1151`（Ryzen AI Max 系）——不包括我们这块。查自己的型号用 `rocminfo | grep gfx`。
2. **系统要求冲突。** AMD 文档写明 Ryzen 平台跑 PyTorch 需要 Ubuntu 24.04 + 6.14 OEM 内核。而我们整套是 Ubuntu 22.04 + ROS2 Humble，换到 24.04 意味着 ROS2 要跟着换代。**为了一块核显把整个机器人的中间件换版本，这笔账怎么算都不划算。**
3. **核显本身算力有限。** APU 的定位就是轻量推理，就算通了也快不到哪去。

所以我们**直接用 CPU 跑 `.pt`**，配置里一行 `device: "cpu"` 解决问题。

坊间有个绕过去的办法，设环境变量骗 ROCm 把 `gfx1103` 当成受支持的型号（`HSA_OVERRIDE_GFX_VERSION=11.0.0`），有人跑通过，但属于非官方、不保证稳定、出问题没人管的状态。

> 什么时候值得回头再看？三条满足两条就重新评估一次：换了 Ryzen AI 300 / Max 系的小电脑；系统本来就要升到 Ubuntu 24.04；或者干脆上一块 NVIDIA 的小卡，那就回到熟路了。
>
> 这个"不做"的决定本身也是想传达的东西：**工程里经常要判断一件事值不值得做，"查清楚为什么不做"和"做出来"一样是有效的工作。** 怕的是既没查清楚，又稀里糊涂折腾了两周。

---

## Y.8 常见问题排查表

| 现象 | 可能原因 | 怎么办 |
| --- | --- | --- |
| `torch.cuda.is_available()` 是 `False` | 装的是 CPU 版 | 看 `torch.__version__` 有没有 `+cpu` |
| 同上 | 驱动 / 容器问题 | 见 CUDA 章的排查表 |
| `pip install torch` 装出来的不是想要的版本 | 没加 `--index-url` | 见 Y.2 |
| 装了 ultralytics 之后变成 CPU 版 torch | 依赖链自动解析拉错了 | 先手动装对 PyTorch，再装 ultralytics |
| 不知道环境到底对不对 | — | 跑 `yolo checks`，一次报全 |
| 第一次运行卡在下载权重 | 在自动下载基础模型 | 等一下；或按 Y.3.2 手动下载 |
| `Weights only load failed` / `Unsupported global` | PyTorch 2.6+ 的 `weights_only` 默认值变了 | `torch.load(..., weights_only=False)`，见 Y.6 |
| 加载权重报缺少某个类 | ultralytics 版本比权重旧 | 对比 `ultralytics.__version__` 和 `ckpt['version']` |
| 换了新版本 YOLO，不报错但结果不对 | API / 参数 / 返回结构变了 | 见 Y.4.4 的四步 |
| 旧权重给新版本用报错 | 权重不能跨版本直接用 | 重新训练 |
| 实时循环里日志疯狂刷屏 | Ultralytics 默认每帧打印 | 加 `verbose=False` |
| 部署效果比训练时差很多 | 拿了 `last.pt` 不是 `best.pt` | 见 Y.6 |
| 以为在用 GPU，实际在用 CPU | 代码静默退回了 | 加警告，见 Y.5.1；或打印 `model.device` |
| 小电脑上多个节点一起跑就都变慢 | 线程抢核 | 限制线程数，见 Y.5.2 |
| CPU 上开了 `.half()` 反而更慢 | CPU 对 FP16 支持不完整 | CPU 上别用半精度，见 Y.5.3 |
| 小电脑上帧率个位数 | CPU 推理本来就慢 | 降尺寸、换小模型、跳帧，见 Y.5.3 |
| 检测不到小目标 | 输入尺寸太小 / 模型太小 | 调大 `imgsz`；换大一号模型 |

---

## Y.9 关键词与命令速查表

### Y.9.1 概念

| 名词 | 一句话说明 |
| --- | --- |
| **PyTorch** | 我们用的深度学习框架，Ultralytics 建立在它之上 |
| **Ultralytics** | YOLO 的官方封装库，提供 `yolo` 命令和 `YOLO` 类 |
| **`.pt` / `.pth`** | PyTorch 的模型文件，本质是个 ZIP 包 |
| **基础模型 / 预训练权重** | 官方在 COCO 上练好的通用模型，用来验证环境和当训练起点 |
| **`best.pt`** | 验证集上表现最好的那一轮，**部署用这个** |
| **`last.pt`** | 最后一轮，用来断点续训 |
| **n / s / m / l / x** | 模型尺寸，从小到大。边缘设备用 n |
| **detect / segment / pose / obb / classify** | 五种任务类型，权重文件名后缀能看出来 |
| **NMS** | 非极大值抑制，把重叠的框合并掉的后处理步骤 |
| **AGPL-3.0** | Ultralytics 的许可证，涉及闭源商用要先搞清楚 |
| **ROCm** | AMD 对标 CUDA 的计算平台 |
| **gfx1103** | AMD 的架构代号，相当于 NVIDIA 的 `sm_xx` |

### Y.9.2 环境与自检

| 写法 | 一句话说明 |
| --- | --- |
| `pip3 install torch --index-url .../cu126` | 装指定 CUDA 版本的 PyTorch，**`--index-url` 不能省** |
| `pip3 install ultralytics` | 装 Ultralytics |
| `yolo checks` | **一次性报全环境状况**，装完先跑这个 |
| `torch.__version__` | 版本字符串，**看 `+cu126` / `+cpu` 后缀** |
| `torch.cuda.is_available()` | 有没有可用 GPU |
| `torch.version.hip` | 不是 `None` 就说明这是 ROCm 版 |
| `ultralytics.__version__` | Ultralytics 版本 |
| `torch.get_num_threads()` | 当前 CPU 线程数 |
| `torch.set_num_threads(n)` | **限制 CPU 线程数**，多节点共存时必设 |
| `cv2.setNumThreads(n)` | OpenCV 的线程数，容易被忘 |
| `OMP_NUM_THREADS` | 环境变量版，要在 `import torch` 之前设 |

### Y.9.3 `yolo` 命令行

| 命令 | 一句话说明 |
| --- | --- |
| `yolo [任务] 模式 参数=值` | 通用语法 |
| `yolo predict model=x.pt source=test.jpg` | 推理一张图 |
| `yolo predict model=x.pt source=0 show=True` | 摄像头实时推理并显示 |
| `yolo predict ... save=True` | 结果存到 `runs/detect/predict/` |
| `yolo export model=x.pt format=onnx` | 导出成 ONNX |
| `yolo val model=x.pt data=data.yaml` | 在验证集上评估 |
| `yolo checks` | 环境自检 |

### Y.9.4 Python API

| 写法 | 一句话说明 |
| --- | --- |
| `YOLO('yolov8n.pt')` | 加载模型，本地没有会自动下载 |
| `model.to(device)` | 把模型搬到指定设备 |
| `model.names` | 类别编号 → 名字的字典 |
| `model.task` | 任务类型（detect / segment / ...） |
| `model.ckpt` | 原始 checkpoint 字典，含 `train_args`、`date`、`version` |
| `model(img, conf=0.5, verbose=False)` | 推理。**实时循环里 `verbose=False` 必加** |
| `results[0].boxes` | 检测框，含 `.xyxy` / `.conf` / `.cls` |
| `results[0].plot()` | 返回画好框的图，可直接 `cv2.imshow` |
| `results[0].speed` | Ultralytics 自报的三段耗时（预处理/推理/后处理） |
| `torch.load(p, weights_only=False)` | 加载完整 checkpoint，**2.6+ 必须显式关掉** |
| `x.to(device)` / `x.detach().cpu().numpy()` | 搬张量 / 取结果 |

### Y.9.5 ROCm 侧（了解即可）

| 命令 | 一句话说明 |
| --- | --- |
| `rocm-smi` | 相当于 `nvidia-smi` |
| `rocminfo \| grep gfx` | 查 GPU 架构代号 |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | 骗 ROCm 把不支持的型号当成支持的，**非官方做法** |

---

## 本章练习

1. 从 PyTorch 官网生成适合你机器的安装命令，装好后打印 `torch.__version__`，确认后缀对不对。
2. 装好 ultralytics，跑 `yolo checks`，把输出贴进笔记。
3. 用命令行方式跑通一次推理（`yolo predict`），找到结果图存在哪。
4. 用 Python 方式跑通一次推理，把每个检测框的类别名、置信度、坐标打印出来。
5. 把第 4 题改成接摄像头的实时循环，确认窗口能弹出、按 `q` 能退出。
6. 手动从 Ultralytics 权重下载页下一个 `yolov8n.pt`，放到脚本旁边，确认不联网也能加载。
7. 下载 `yolov8n.pt` 和 `yolov8s.pt`，比较文件大小和推理耗时，说明尺寸选择的代价。
8. 下载一个 `-seg` 权重跑一次，看看输出和普通检测有什么不一样。
9. 用 Y.6 的方法打开仓库里的 `best.pt` 和 `kfs.pt`，记下它们的类别、基础模型、输入尺寸。
10. 对比 `ultralytics.__version__` 和这两个权重的 `ckpt['version']`，判断有没有版本风险，并在小电脑上实际验证能不能正常加载。
11. 把 `pick_device()` 接进 `detector_node.py`，让配置支持 `device: "auto"`。
12. 在小电脑上测推理耗时：先不限制线程，再设 `torch.set_num_threads(4)`，同时开着其他 ROS2 节点，比较两次结果。
13. 在小电脑上把 `imgsz` 从 640 改到 480 和 320，记录帧率变化，观察检测效果差了多少。
14. 进阶：在 CPU 上分别用 FP32 和 `.half()` 测一次速度，验证本章说的"CPU 上半精度更慢"。

---

## 本章小结

这一章能压成四块：

```text
   装环境：
     PyTorch 版本字符串的后缀（+cu126 / +cpu）决定一切
     先手动装对 PyTorch，再装 ultralytics
     yolo checks 一次报全

   跑通：
     基础模型第一次运行自动下载
     命令行 yolo predict 验证最快
     实时循环记得 verbose=False

   选型：
     任务类型 → 模型尺寸 → 模型版本，按这个顺序定
     一律从 n 起步；能用检测就别用分割
     不是越新越好：资料量、导出链路、API 稳定性都要算进去

   两台机器：
     device 写 auto，不硬编码
     CPU 上要限线程，不要用 half
     退回 CPU 时一定要吵一声
```

三个最容易犯的错，避开就算这一章到位了：

- **装了 CPU 版还在找为什么用不了 GPU。** 先看版本字符串。
- **选型只看精度指标。** 生态成熟度、导出链路、团队现状，在工程里往往比 mAP 高零点几更重要。
- **静默退回 CPU。** 你以为在用显卡，其实没有，然后对着帧率发愁。

---

## 参考资料

- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)（装之前先来这里）
- [Ultralytics 官方文档](https://docs.ultralytics.com/)（中文版有语言切换）
- [Ultralytics 支持的模型列表](https://docs.ultralytics.com/models/)（选型时对照着看）
- [Ultralytics 预训练权重下载](https://github.com/ultralytics/assets/releases)（手动下载基础模型）
- [Ultralytics 推理模式文档](https://docs.ultralytics.com/modes/predict/)（`predict` 的全部参数）
- [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)（ONNX / TensorRT / OpenVINO）
- [ROCm 兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)（查自己的 GPU 在不在支持列表里）
- 驱动、CUDA、测速的完整流程：本教材 CUDA 入门那一章
- 容器里怎么装这一套：本教材 6 章
- 数据集、训练、评估：本教材有监督学习那一章
