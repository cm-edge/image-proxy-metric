from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union


# ============================================================
# 0) Type definitions
# ============================================================

# Detection format:
# (left, top, width, height, confidence, class_name)
Detection = Tuple[float, float, float, float, float, str]


@dataclass
class MatchDetail:
    """
    记录 curr frame 中一个 detection 的匹配情况。
    """
    curr_index: int
    prev_index: int   # 如果没有匹配上，则为 -1
    best_iou: float
    matched: bool


# ============================================================
# 1) IoU between two boxes
# ============================================================

def _bbox_iou_xywh(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    计算两个 bbox 的 IoU。

    Parameters
    ----------
    box1, box2:
        格式均为 (left, top, width, height)

    Returns
    -------
    float
        IoU in [0, 1]
    """
    x1_1, y1_1, w1, h1 = box1
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    # intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    # union
    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union = area1 + area2 - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


# ============================================================
# 2) Find best match for one current detection
# ============================================================

def _find_best_match_for_one_curr_det(
    curr_det: Detection,
    prev_dets: List[Detection],
    used_prev: List[bool],
) -> Tuple[int, float]:
    """
    对当前帧中的一个 detection，在上一帧中寻找最佳匹配。

    匹配约束：
    1. previous detection 还没有被占用
    2. class label 必须相同
    3. 在满足前两条的候选里，取 IoU 最大者

    Returns
    -------
    best_prev_index : int
        最佳匹配在 prev_dets 中的下标；若不存在则为 -1
    best_iou : float
        对应的最大 IoU；若不存在则为 0.0
    """
    l2, t2, w2, h2, conf2, cls2 = curr_det

    best_prev_index = -1
    best_iou = 0.0

    for j, prev_det in enumerate(prev_dets):
        if used_prev[j]:
            continue

        l1, t1, w1, h1, conf1, cls1 = prev_det

        # 类别不同，不允许匹配
        if cls1 != cls2:
            continue

        iou = _bbox_iou_xywh((l1, t1, w1, h1), (l2, t2, w2, h2))

        if iou > best_iou:
            best_iou = iou
            best_prev_index = j

    return best_prev_index, best_iou


# ============================================================
# 3) Atomic PTC-IoU between two frames
# ============================================================

def compute_ptc_iou_pair(
    prev_dets: List[Detection],
    curr_dets: List[Detection],
    iou_thr: float = 0.5,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    计算两个相邻帧之间的 PTC-IoU。

    Definition
    ----------
    给定：
      - prev_dets: frame t-1 的 detections
      - curr_dets: frame t 的 detections

    对 curr_dets 中的每一个 detection：
      - 在 prev_dets 中找一个“尚未使用、类别相同、IoU 最大”的框
      - 若该最大 IoU >= iou_thr，则认为这个 curr detection 被成功匹配

    最终定义为：

        PTC_IoU = matched_count / len(curr_dets)

    也就是说，它衡量的是：
    “当前帧中，有多少检测框能在前一帧里找到时间上一致的支持”。

    Parameters
    ----------
    prev_dets, curr_dets:
        detection list，单个 detection 格式为
        (left, top, width, height, confidence, class_name)

    iou_thr:
        判定匹配成功所需的最小 IoU 阈值

    return_details:
        True 时返回详细匹配信息，便于分析这个 proxy 是否合理

    Boundary cases
    --------------
    - prev = [] 且 curr = []  -> 1.0
    - prev != [] 且 curr = [] -> 0.0
    - prev = [] 且 curr != [] -> 0.0

    Returns
    -------
    float
        当 return_details=False 时
    dict
        当 return_details=True 时，返回 score 和每个 curr det 的匹配细节
    """
    prev_dets = prev_dets or []
    curr_dets = curr_dets or []

    # 情况 1：两帧都没有检测框
    if len(prev_dets) == 0 and len(curr_dets) == 0:
        score = 1.0
        if not return_details:
            return score
        return {
            "score": score,
            "matched_count": 0,
            "num_prev": 0,
            "num_curr": 0,
            "details": [],
        }

    # 情况 2：当前帧没有检测框
    if len(curr_dets) == 0:
        score = 0.0
        if not return_details:
            return score
        return {
            "score": score,
            "matched_count": 0,
            "num_prev": len(prev_dets),
            "num_curr": 0,
            "details": [],
        }

    used_prev = [False] * len(prev_dets)
    matched_count = 0
    details: List[MatchDetail] = []

    # 逐个处理当前帧中的 detection
    for curr_index, curr_det in enumerate(curr_dets):
        best_prev_index, best_iou = _find_best_match_for_one_curr_det(
            curr_det=curr_det,
            prev_dets=prev_dets,
            used_prev=used_prev,
        )

        # 只有 IoU 达到阈值，才认为匹配成功
        if best_prev_index >= 0 and best_iou >= iou_thr:
            used_prev[best_prev_index] = True
            matched_count += 1
            details.append(
                MatchDetail(
                    curr_index=curr_index,
                    prev_index=best_prev_index,
                    best_iou=best_iou,
                    matched=True,
                )
            )
        else:
            details.append(
                MatchDetail(
                    curr_index=curr_index,
                    prev_index=-1,
                    best_iou=best_iou,
                    matched=False,
                )
            )

    score = matched_count / float(len(curr_dets))

    if not return_details:
        return score

    return {
        "score": score,
        "matched_count": matched_count,
        "num_prev": len(prev_dets),
        "num_curr": len(curr_dets),
        "details": details,
    }