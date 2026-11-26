# YOLOv8 项目资料归档说明

本归档包含 Robocon 视觉组两个 YOLOv8 实例分割项目的**教学文档、训练数据集、模型评估文件**。

## 一、目录结构

```
yolov8模型训练/
├── 5 份教学文档（根目录）
├── 番茄成熟度识别模型/
│   ├── 数据集/
│   │   ├── data.yaml（可移植版）+ data_原始绝对路径版.yaml
│   │   ├── train/（images 676 + labels 676）
│   │   ├── val/（images 118 + labels 118）
│   │   └── test/（images 55 + labels 55）
│   └── 模型评估文件/（最终模型 tomato_seg-7，含 weights/best.pt）
└── 玉米大斑病检测模型/
    ├── 数据集/
    │   ├── data.yaml（可移植版）+ data_原始绝对路径版.yaml
    │   ├── images/（train 374 + val 93 + test 90，jpg 各配同名 json）
    │   └── labels/（train 374 + val 93 + test 90，txt）
    └── 模型评估文件/（最终模型 maize_seg-2，含 weights/best.pt、README、edge_contour、test_predict_grid）
```

## 二、数据集口径

| 项目 | 类别 | 训练/验证/测试 | 图片总数 |
|---|---|---|---|
| 番茄成熟度 | tomato_R(0 成熟) / tomato_G(1 未成熟) | 676 / 118 / 55 | 849 |
| 玉米大斑病 | maize_01(0 健康) / maize_02(1 大斑病) | 374 / 93 / 90 | 557 |

- 番茄标注对象是**果实**，玉米标注对象是**整株植株**（polygon）；
- 已排除可自动重建的 `labels.cache`；
- 番茄原始素材（Kaggle 目录、base.zip 等）未纳入，仅保留最终训练集。

## 三、模型评估文件版本说明

归档放的是**两个项目的最终模型**：

| 项目 | 归档版本 | 关键指标 |
|---|---|---|
| 番茄 | tomato_seg-7（YOLOv8n-seg，epoch 81） | Box mAP50 0.836 / Mask mAP50 0.834 |
| 玉米 | maize_seg-2 | val Mask mAP50 0.977 / 全新场景 test 0.956 |

> 早期版本（番茄 seg-6、玉米 seg-1）的评估包未纳入本归档；如需保留历史版本，可另行补充。

## 四、路径可移植性（重要）

- 原始 `data.yaml` 写有作者机器的绝对路径（`C:/...`、`D:/...`），已原样保留为 `data_原始绝对路径版.yaml`；
- 同目录另附可移植版 `data.yaml`，使用相对路径 `path: .`，换机器直接可用；
- 训练时在对应 `data.yaml` 所在目录运行，或把 `path` 改成你自己的绝对路径。

## 五、使用环境

```
Ultralytics 8.4.x + PyTorch 2.5.1+cu121 + Python 3.12
```

`best.pt` 为标准 YOLO 权重，不绑定训练环境，8.4.x 各小版本均可加载。
