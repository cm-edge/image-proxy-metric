# image-proxy-metric

本仓库包含一组用于**无监督（GT-free）评估目标检测结果稳定性/一致性**的代理指标（proxy metrics）工具。

当前包含两个主要工具：

- `tta_locu_tool/` —— TTA Loc-U（基于 Test-Time Augmentation 的定位不确定性度量）
- `ptc_iou_tool/` —— PTC-IoU（基于相邻帧 IoU 的时间一致性指标）
- `shared/` —— 公共模块（检测器封装、通用函数等）
- `test_run.py` —— 本地测试入口

> 注意：本仓库中的指标为“代理指标（proxy metrics）”，不依赖 Ground Truth，不用于衡量检测准确率。

---

# 1. 仓库结构

```text
image-proxy-metric/
├─ ptc_iou_tool/
├─ shared/
├─ tta_locu_tool/
├─ test_run.py
└─ README.md
```

# 2. 运行环境
- Python >= 3.8（根据你的实际版本修改）
- 若使用 GPU 检测模型，需安装对应 CUDA

# 3. 工具说明
## 3.1 TTA Loc-U
### 功能
TTA Loc-U 用于衡量目标检测模型在 Test-Time Augmentation (TTA) 下的定位不确定性。

核心思想：
- 若同一目标在多次 TTA 推理后预测框高度一致 → 定位稳定 → 不确定性低
- 若预测框分散 → 定位不稳定 → 不确定性高

该指标为 GT-free 指标。

## 3.2 PTC-IoU
### 功能
PTC-IoU 用于评估连续图像序列中目标检测结果的时间一致性。

核心思想：
- 相邻帧之间，同类别检测框若存在高 IoU 匹配 → 说明预测稳定
- 若频繁抖动或丢失 → PTC-IoU 值降低

该指标同样为 GT-free 指标。

# 4. ignored_region 说明
支持读取序列级 XML 中的 ignored_region，用于过滤检测结果。

说明：
- XML 仅用于 ignored_region 过滤
- 不使用任何 Ground Truth 标注

# 5. 注意事项
- 本仓库仍在开发中，接口可能调整
- 不同检测模型需实现统一 Detector 接口（位于 shared/）