# PTC-IoU：无 Ground Truth 的目标检测时间一致性评估工具

本工具用于评估目标检测模型在连续图像序列上的预测时间一致性，计算逐帧的 PTC_iou（Prediction Time Consistency IoU）分数。

## 1. Method Overview
PTC-IoU 衡量两个相邻帧之间检测结果的一致程度
其核心思想是：若目标在时间上稳定，则相邻帧之间应存在高 IoU 且类别一致的检测匹配


## 2. Method Definition
### 2.1 Atomic Definition
PTC-IoU 定义在两个相邻帧之间：

```text
PTC_t = (# matched detections) / (# detections in frame t)
```

匹配规则如下：
- 检测类别相同
- IoU ≥ 指定阈值
- 一对一匹配

代码实现如下：

```python
compute_ptc_iou_pair(prev_dets, curr_dets)
```

其中：
- prev_dets 表示上一帧的检测结果
- curr_dets 表示当前帧的检测结果
- 检测格式为 (left, top, width, height, conf, class_name)

该函数是方法的原子定义

### 2.2 Image-Level Wrapper (Two Images)
为了方便使用，工具提供图像级封装函数：

```python
compute_ptc_iou_pair_from_images(
    prev_img_path,
    curr_img_path,
    model_name,
    conf_threshold=0.3,
    iou_thr=0.5
)
```

该函数内部：
- 调用指定检测模型
- 获得两帧检测结果
- 调用 atomic metric
- 返回单个 PTC-IoU 值

### 2.3 Sequence-Level Definition
整段序列的 PTC 定义为相邻帧 atomic metric 的堆叠结果：
```text
PTC_series[t] = PTC(frame_{t-1}, frame_t)
```
可选时间平滑处理。

对应函数：
```python
compute_ptc_iou_series(dets_all)
```

## 3. Installation

推荐使用 Python 3.9 及以上版本。

在项目根目录运行：
```bash
pip install -e .
```

安装完成后将自动安装以下依赖：
- numpy
- torch
- torchvision
- pillow

## 4 Input Data Structure (Sequence-Level Mode)

### 4.1 Image Directory (Required)

```text
IMG_ROOT/
  ├─ SEQ_1/
  │    ├─ img00001.jpg
  │    ├─ img00002.jpg
  │    └─ ...
  ├─ SEQ_2/
  └─ ...
  ```
每个子文件夹表示一个图像序列。

### 4.2 XML Directory (Optional)
```text
SEQ_XML_ROOT/
  ├─ SEQ_1.xml
  ├─ SEQ_2.xml
  └─ ...
```
XML 文件与图像序列同名，仅用于读取 ignored_region。

## 5. Usage
### 5.1 Command Line Interface
安装完成后可直接使用命令行工具：
```bash
ptc-iou \
  --img_root "/path/to/images" \
  --seq_xml_root "/path/to/xml" \
  --out_root "/path/to/output" \
  --models fasterrcnn_mobilenet_v3_large_fpn \
  --confs 0.5 \
  --iou_thr 0.5 \
  --smooth 3
```
调试模式：
```bash
ptc-iou \
  --img_root "/path/to/images" \
  --seq_xml_root "/path/to/xml" \
  --out_root "/path/to/output" \
  --models fasterrcnn_mobilenet_v3_large_fpn \
  --confs 0.5 \
  --seqs MVI_39031 \
  --max_frames 200
```

### 5.2 Python API
Atomic (Two Detection Lists)
```python
from ptc_iou_tool import compute_ptc_iou_pair

score = compute_ptc_iou_pair(prev_dets, curr_dets)
```

Image-Level (Two Images)
```python
from ptc_iou_tool import compute_ptc_iou_pair_from_images

score = compute_ptc_iou_pair_from_images(
    "img0001.jpg",
    "img0002.jpg",
    model_name="fasterrcnn_mobilenet_v3_large_fpn"
)
```

Sequence-Level
```python
from ptc_iou_tool import compute_ptc_iou_series

ptc_series = compute_ptc_iou_series(dets_all)
```

Run Entire Dataset
```python
from ptc_iou_tool import run_ptc_iou_on_img_root

run_ptc_iou_on_img_root(
    img_root="...",
    out_root="...",
    models=["fasterrcnn_mobilenet_v3_large_fpn"],
    confs=[0.5],
)
```

## 6. Output
对于每个 sequence + model + confidence 组合，将生成：

- .npy 文件：逐帧 PTC 数组
- .csv 文件：逐帧 PTC 与图像路径对应表

## 7. Characteristics
- GT-free；
- 局部时间一致性度量
- 原子化定义（两帧级别）
- 可扩展至任意检测模型
- 可用于稳定性分析、抖动分析与伪标签筛选