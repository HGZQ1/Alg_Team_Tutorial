> **插入说明**：这一节放在原来的「常见问题排查表」之前，也就是接在 X.9 Jetson 那一节后面。插进去之后记得把后面两节顺延：
> - `## X.10 常见问题排查表` → `## X.11 常见问题排查表`
> - `## X.11 关键词与 API 速查表` → `## X.12 关键词与 API 速查表`（连同它下面的 `X.11.1`~`X.11.4` 一起改成 `X.12.x`）
>
> 另外章首那句「在 **X.10 速查表**里都有一句话解释」原本编号就写错了（速查表当时是 X.11），一并改成 **X.12** 就对了。

---

## X.10 实战：从零跑通一次 YOLO 推理并测速

前面讲的都是零件，这一节把它们装成一台能跑的机器。目标是：**从一张图开始，一步步做到知道自己每个环节花了多少毫秒。**

跟着做完，你会得到两个能直接用的脚本，以后换模型、换机器都能拿来测。

### X.10.1 准备

```bash
pip3 install ultralytics
```

Ultralytics 会把 PyTorch 一起带上。装完先确认一下 GPU 是通的：

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

模型权重第一次运行会自动下载（`yolov8n.pt` 只有几 MB）。如果机器没网，提前在别处下好放到脚本旁边就行。

> 用 `n`（nano）这个最小的版本起步。等流程跑通了再换 `s`、`m`，能很直观地看到模型大小对速度的影响。

### X.10.2 第一步：先让它跑起来

不管性能，五行代码：

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('test.jpg')
results[0].show()          # 弹窗看结果
print(results[0].boxes)    # 打印检测框
```

能看到画着框的图，第一步就成了。

**这时候先别急着测速度。** 先确认它真的用上了 GPU：

```python
model = YOLO('yolov8n.pt')
model.to('cuda')
print(next(model.model.parameters()).device)     # 应该打印 cuda:0
```

如果这里是 `cpu`，后面测什么都没意义。

### X.10.3 第二步：加上正确的测速

Ultralytics 自己就会报速度，藏在结果里：

```python
results = model('test.jpg')
print(results[0].speed)
# {'preprocess': 1.8, 'inference': 2.3, 'postprocess': 0.9}   单位是毫秒
```

这三个数字很有用，但**第一次跑出来的数字是不能信的**——冷启动没预热，而且它内部有没有正确同步我们也不好确认。

所以还是要自己测一遍。下面是纯推理部分的测速，直接测底层的 `nn.Module`，排除掉预处理和后处理的干扰：

```python
import time
import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8n.pt')
net = model.model.to(device).eval()          # 拿到底层的 PyTorch 模型

x = torch.randn(1, 3, 640, 640, device=device)   # 假数据，尺寸和真实输入一致

with torch.no_grad():
    # 预热
    for _ in range(30):
        net(x)
    torch.cuda.synchronize()

    # 正式计时
    t0 = time.time()
    for _ in range(200):
        net(x)
    torch.cuda.synchronize()
    dt = time.time() - t0

print(f"纯推理 {dt/200*1000:.2f} ms  |  {200/dt:.0f} FPS")
```

三个必须有的东西，少一个数据就废了：

| 要素 | 少了会怎样 |
| --- | --- |
| `net.eval()` | BatchNorm 行为不对，结果错但不报错 |
| `torch.no_grad()` | 会建计算图，测出来偏慢，还多占显存 |
| `torch.cuda.synchronize()` | 测的是提交任务的时间，FPS 高得离谱 |

### X.10.4 第三步：FP32 和 FP16 对比

改两行就能测半精度：

```python
net = net.half()
x = x.half()
```

在我们的开发机（RTX 5080 Laptop）上，`yolov8n` 640×640 的大致结果是：

| 精度 | 单次推理 | 纯推理 FPS |
| --- | --- | --- |
| FP32 | 约 2.5 ms | 约 400 |
| FP16 | 约 1.3 ms | 约 750 |

**你的数字一定和这个不一样**，显卡型号、驱动、PyTorch 版本、功耗模式都会影响。重点不是对上这个数，而是**FP16 大致能快一倍**这个量级。

顺便验证一下精度损失有多大：

```python
with torch.no_grad():
    out32 = net.float()(x.float())[0]
    out16 = net.half()(x.half())[0]
    diff = (out32 - out16.float()).abs().max().item()
print(f"最大差异 {diff:.6f}")
```

对检测任务来说，这个差异通常小到框的位置差不了半个像素。但**换个模型就要重新验一次**，别默认它一定没事。

### X.10.5 第四步：完整的测速脚本

把上面的东西整理成一个能直接跑的文件。这个脚本会自检环境、测 FP32/FP16、并对比不同输入尺寸：

```python
# bench_yolo.py
import time
import torch
from ultralytics import YOLO


def check_env():
    print("=" * 50)
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA 可用    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("没有可用的 GPU，后面的测试没有意义，先解决环境问题")
        return False
    print(f"显卡         : {torch.cuda.get_device_name()}")
    print(f"算力         : sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"torch 支持   : {torch.cuda.get_arch_list()}")
    print(f"CUDA 运行时  : {torch.version.cuda}")
    print("=" * 50)

    # 真的算一次，光看 is_available 不算数
    try:
        a = torch.randn(512, 512, device='cuda')
        (a @ a).sum().item()
        print("矩阵乘法通过，环境正常\n")
        return True
    except Exception as e:
        print(f"能初始化但算不了，多半是版本对不上：\n{e}")
        return False


def bench(net, size, half, warmup=30, runs=200):
    device = 'cuda'
    net = net.half() if half else net.float()
    x = torch.randn(1, 3, size, size, device=device)
    x = x.half() if half else x.float()

    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(runs):
            net(x)
        torch.cuda.synchronize()
        dt = time.time() - t0

    ms = dt / runs * 1000
    mem = torch.cuda.max_memory_allocated() / 1024**2
    return ms, runs / dt, mem


def main():
    if not check_env():
        return

    model = YOLO('yolov8n.pt')
    net = model.model.to('cuda').eval()

    print(f"{'尺寸':>6} {'精度':>6} {'耗时(ms)':>10} {'FPS':>8} {'峰值显存(MB)':>14}")
    print("-" * 50)

    for size in (320, 640, 960):
        for half in (False, True):
            torch.cuda.reset_peak_memory_stats()
            ms, fps, mem = bench(net, size, half)
            print(f"{size:>6} {'FP16' if half else 'FP32':>6} "
                  f"{ms:>10.2f} {fps:>8.0f} {mem:>14.1f}")


if __name__ == '__main__':
    main()
```

跑出来大概长这样：

```text
  尺寸   精度   耗时(ms)      FPS   峰值显存(MB)
--------------------------------------------------
   320   FP32       0.98     1020           82.4
   320   FP16       0.61     1639           47.1
   640   FP32       2.51      398          210.6
   640   FP16       1.32      758          118.3
   960   FP32       5.43      184          452.9
   960   FP16       2.87      348          251.7
```

从这张表能读出三件事，比背结论有用得多：

- **输入尺寸翻倍，耗时大约变成四倍**——因为像素数是平方关系；
- **FP16 稳定快一倍左右**，显存也差不多减半；
- 320 输入能跑到一千多 FPS，**但这没什么意义**，因为相机根本给不了这么多帧。到这一步瓶颈已经彻底不在 GPU 了。

最后一条特别值得记住：**优化到某个点之后，继续榨 GPU 是没有收益的**，该去看别的环节了。

### X.10.6 第五步：接上摄像头，分段计时

离线测的是理论上限。真实管线里还有读帧、预处理、后处理、显示，这些往往才是大头。

```python
# realtime_yolo.py
import time
import cv2
import torch
from ultralytics import YOLO

CAM_INDEX = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8n.pt')
model.to(device)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 预热：拿假图先跑几十遍，别让第一帧背锅
dummy = torch.zeros(480, 640, 3, dtype=torch.uint8).numpy()
for _ in range(20):
    model.predict(dummy, device=device, half=True, verbose=False)
if device == 'cuda':
    torch.cuda.synchronize()
print("预热完成")


def stamp():
    """带同步的时间戳，专门用来分段计时"""
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.time()


while True:
    t0 = stamp()
    ret, frame = cap.read()
    if not ret:
        break
    t1 = stamp()

    results = model.predict(frame, device=device, half=True, verbose=False)
    t2 = stamp()

    annotated = results[0].plot()
    t3 = stamp()

    # 各段耗时（毫秒）
    read_ms  = (t1 - t0) * 1000
    infer_ms = (t2 - t1) * 1000
    draw_ms  = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    # ultralytics 自己报的内部拆分
    sp = results[0].speed

    cv2.putText(annotated, f"total {total_ms:5.1f}ms  {1000/total_ms:5.1f} FPS",
                (10, 30), 1, 1.4, (0, 255, 0), 2)
    cv2.putText(annotated, f"read {read_ms:5.1f}  infer {infer_ms:5.1f}  draw {draw_ms:5.1f}",
                (10, 60), 1, 1.2, (255, 0, 255), 2)
    cv2.putText(annotated,
                f"pre {sp['preprocess']:.1f}  net {sp['inference']:.1f}  post {sp['postprocess']:.1f}",
                (10, 90), 1, 1.2, (0, 255, 255), 2)

    cv2.imshow('yolo', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

两个细节：

- 那个 `stamp()` 函数把 `synchronize()` 包了进去。分段计时的每一个时间点都必须同步，否则各段的数字会互相串味；
- 预热用的是一张全黑的假图，尺寸和真实输入一致。**开机预热这一步在实际部署时可以有**，避免启动的头几帧会明显卡顿。

### X.10.7 读懂你测出来的数字

跑起来之后大概率会看到类似这样的画面：

```text
   total  33.5ms   29.9 FPS
   read  28.1  infer   4.2  draw   1.2
   pre 1.5  net 2.3  post 0.4
```

**这组数字在说什么？** 推理只花了 4.2ms，但读一帧花了 28.1ms，整体被死死卡在 30 FPS。

原因是 `cap.read()` 是**阻塞**的——相机只有 30fps，它就得在那儿等下一帧到来。这不是相机慢，是你的程序已经比相机快太多了。如果想量模型真实上限，应该拿同一张图连续推理 100 次取平均。

常见几种情况的对号入座：

| 你看到的 | 说明 | 该做什么 |
| --- | --- | --- |
| `read` 占了绝大部分 | 程序比相机快，被相机限速 | 提高相机帧率（MJPG、降分辨率），或者接受现状 |
| `infer` 占大头 | GPU 真的忙不过来 | 开 FP16、换小模型、降输入尺寸、上 TensorRT |
| `post` 占大头 | NMS 或后处理慢 | 提高置信度阈值减少候选框；检查后处理里有没有 Python 循环 |
| `draw` 占大头 | 画框和 `imshow` 慢 | 调试完就把可视化关掉，正式跑不需要它 |
| 各段都不大但 total 大 | 中间有没测到的东西 | 补测漏掉的环节，比如串口发送、话题发布 |

最后一行那句"**读帧 28ms**"其实是好消息——它意味着你还有 24ms 的余量可以拿去做别的事情（比如再跑一个模型、做点滤波、发几个话题）。当然在硬件允许的情况下也可以选择提高采样帧绿。

> 一个很实际的建议：**把这套分段计时留在代码里，用一个开关控制**。等到比赛前两天发现帧率掉了，你会庆幸自己不用临时去加计时代码。

### X.10.8 可以补进本章练习

下面几道可以直接续在原来的练习后面：

1.  跑通 `bench_yolo.py`，把你机器上的那张表贴进笔记，和同学的对比一下。
2.  把 `yolov8n` 换成 `yolov8s` 和 `yolov8m`，看模型大小和耗时是什么关系，是线性吗？
3.  把 `bench_yolo.py` 里的 `warmup` 改成 0，看结果变化有多大，解释为什么。
4.  跑 `realtime_yolo.py`，找出你这台机器上的瓶颈在哪一段，并写下你打算怎么优化。
5.  把 `realtime_yolo.py` 里的可视化（`plot` 和 `imshow`）注释掉，看整体 FPS 变了多少。
6.  进阶：把读帧放进一个独立线程，让它和推理并行，看能不能突破相机帧率带来的限制。
