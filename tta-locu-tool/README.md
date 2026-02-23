# TTA Loc-U 工具

本工具用于评估目标检测模型在 Test-Time Augmentation (TTA) 下的定位不确定性，计算逐帧 Loc-U（Localization Uncertainty）分数。



## 1. Method Overview

Loc-U 衡量在多次 TTA 推理下，同一目标预测框的空间不确定性。

其核心思想是：

- 若目标定位稳定，则多次增强预测应聚集；
- 若定位不稳定，则预测框分布离散。

Loc-U 是一个 GT-free 不确定性指标，仅依赖模型自身预测。

## 2.Method Definition
### 2.1 Atomic Definition
Loc-U 定义在 单帧 + 多次 TTA 推理结果 上。
```text
LocU = mean( GaussianEntropy(cluster_i) )                                          
```

计算流程：

1.对同一帧进行多次 TTA 推理

2.聚合所有预测框

3.NMS 生成 cluster centers

4.基于 IoU 将预测分配到 cluster

5.过滤覆盖率不足的 cluster

6.对每个 cluster 计算归一化 bbox 的协方差

7.计算 Gaussian entropy

8.取平均作为该帧 Loc-U

代码实现如下：
```python
compute_loc_u_for_frame_fixed(dets_each_run)
```

其中：

- dets_each_run 为 List[List[Det]]

- Det 格式为 (left, top, width, height, conf, class_name)

该函数是方法的 原子定义。

### 2.2 Image-Level Wrapper (Single Image + TTA)
为了方便使用，工具提供图像级封装函数：
```python
compute_locu_for_image(
    img,
    predict_fn,
    tta_mode="grid"
)
```

该函数内部：

- 生成 TTA 图像

- 调用检测模型

- 调用 atomic Loc-U

- 返回单帧 Loc-U

### 2.3 Sequence-Level Definition
整段序列的 Loc-U 定义为逐帧 Loc-U 的堆叠结果：
```text
LocU_series[t] = LocU(frame_t)
```
对应函数：
```python
run_locu_on_sequence(...)
```

### 2.4 Dataset-Level Definition
对整个数据集的多个序列与模型组合进行批量计算：
```python
run_locu_on_img_root(...)
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
- opencv-python

## 4. Input Data Structure (Sequence-Level Mode)
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
每个子文件夹表示一个视频序列。

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
tta-locu \
  --img_root "/path/to/images" \
  --seq_xml_root "/path/to/xml" \
  --out_root "/path/to/output" \
  --models fasterrcnn_mobilenet_v3_large_fpn \
  --tta_mode grid \
  --n_runs 10 \
  --sel_thr 0.22 \
  --iou_loc 0.22 \
  --nms_iou 0.45 \
  --min_cov 9
```
调试模式：
```bash
tta-locu \
  --img_root "/path/to/images" \
  --seq_xml_root "/path/to/xml" \
  --out_root "/path/to/output" \
  --models fasterrcnn_mobilenet_v3_large_fpn \
  --seqs MVI_39031 \
  --max_frames 200
```

### 5.2 Python API
Atomic (Single Frame + TTA Results)
```python
from tta_locu_tool import compute_loc_u_for_frame_fixed

locu, n_centers, n_kept = compute_loc_u_for_frame_fixed(dets_each_run)
```

Image-Level
```python
from tta_locu_tool import compute_locu_for_image

locu, n_centers, n_kept = compute_locu_for_image(
    img,
    predict_fn,
    tta_mode="grid"
)
```
Sequence-Level
```python
from tta_locu_tool import run_locu_on_sequence

run_locu_on_sequence(
    seq_name="SEQ_1",
    img_dir="...",
    seq_xml_root="...",
    out_root="...",
    model_name="fasterrcnn_mobilenet_v3_large_fpn",
)
```
Run Entire Dataset
```python
from tta_locu_tool import run_locu_on_img_root

run_locu_on_img_root(
    img_root="...",
    seq_xml_root="...",
    out_root="...",
    models=["fasterrcnn_mobilenet_v3_large_fpn"],
)
```

## 6. Output
对于每个 sequence + model 组合，将生成：

.npy 文件：
- 逐帧 Loc-U 数组
- cluster center 数量
- 保留 cluster 数量

.csv 文件：
- frame_index
- image_name
- loc_u
- num_centers
- num_kept_clusters

## 7. Characteristics
- GT-free
- 基于 TTA 的定位不确定性度量
- 原子化定义（单帧级别）
- 可扩展至任意检测模型
- 可用于：
  - 稳定性分析
  - 定位抖动分析
  - 伪标签筛选
  - 不确定性驱动数据过滤