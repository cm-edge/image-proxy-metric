import os
import glob
import csv
from typing import List, Tuple, Optional, Dict, Any
import xml.etree.ElementTree as ET

import numpy as np

from shared.det_shared.detector import Detector


# ============================================================
# 0) ignored_region
# ============================================================

def load_ignored_regions_from_sequence_xml(
    seq_xml_path: str
) -> List[Tuple[float, float, float, float]]:
    """
    读取 DETRAC 序列级 XML 里的 ignored_region
    返回: List[(left, top, width, height)]
    """
    if (not seq_xml_path) or (not os.path.isfile(seq_xml_path)):
        return []

    tree = ET.parse(seq_xml_path)
    root = tree.getroot()

    ignored = []
    ign_node = root.find("ignored_region")
    if ign_node is None:
        return ignored

    for box in ign_node.findall("box"):
        left = float(box.get("left", "0"))
        top = float(box.get("top", "0"))
        width = float(box.get("width", "0"))
        height = float(box.get("height", "0"))
        ignored.append((left, top, width, height))

    return ignored


def _xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return (x, y, x + w, y + h)


def _box_center_in_region(det_xyxy, region_xyxy):
    x1, y1, x2, y2 = det_xyxy
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    rx1, ry1, rx2, ry2 = region_xyxy
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def filter_detections_by_ignored_regions_center(
    detections: List[Tuple[float, float, float, float, float, str]],
    ignored_regions_xywh: List[Tuple[float, float, float, float]],
):
    """
    过滤规则：预测框中心点落在 ignored_region 内 => 丢弃
    """
    if not ignored_regions_xywh:
        return detections

    ignored_xyxy = [_xywh_to_xyxy(r) for r in ignored_regions_xywh]

    kept = []
    for (l, t, w, h, conf, cls) in detections:
        det_xyxy = _xywh_to_xyxy((l, t, w, h))

        drop = False
        for reg_xyxy in ignored_xyxy:
            if _box_center_in_region(det_xyxy, reg_xyxy):
                drop = True
                break

        if not drop:
            kept.append((l, t, w, h, conf, cls))

    return kept


# ============================================================
# 1) Frame loader
# ============================================================

def load_frame_paths_from_dir(dir_path: str) -> List[str]:
    """
    从文件夹读取所有 .jpg / .png / .jpeg，按文件名排序。只返回 paths（不读图）
    """
    exts = ("*.jpg", "*.png", "*.jpeg")
    frame_paths: List[str] = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(frame_paths)


# ============================================================
# 2) Prediction wrapper
# ============================================================

def make_predict_fn_dets_only(
    model_name: str,
    conf_threshold: float = 0.3,
    ignored_regions_xywh: Optional[List[Tuple[float, float, float, float]]] = None,
):
    detector = Detector(model_name, conf_threshold)

    def predict_fn(img_path: str):
        """
        输入: 图片路径
        输出: dets_out = List[(l, t, w, h, conf, class_name)]
        """
        detections = detector.detect(img_path)

        dets_out = []
        for left, top, width, height, conf, class_name in detections:
            dets_out.append(
                (float(left), float(top), float(width), float(height),
                 float(conf), str(class_name))
            )

        if ignored_regions_xywh:
            dets_out = filter_detections_by_ignored_regions_center(
                dets_out, ignored_regions_xywh
            )

        return dets_out

    return predict_fn


# ============================================================
# 3) IoU
# ============================================================

def _bbox_iou_xywh(b1, b2) -> float:
    """
    b1, b2: (left, top, width, height)
    """
    x1_1, y1_1, w1, h1 = b1
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1

    x1_2, y1_2, w2, h2 = b2
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


# ============================================================
# 4) 原子函数：两个帧之间的 PTC_iou（输入 dets）
# ============================================================

def compute_ptc_iou_pair(
    prev_dets: List[Tuple],
    curr_dets: List[Tuple],
    iou_thr: float = 0.5,
) -> float:
    """
    原子定义：
      PTC_iou(frame_{t-1}, frame_t)

    输入 det 格式:
      (l, t, w, h, conf, cls)

    输出:
      float in [0, 1]
    """
    prev_dets = prev_dets or []
    curr_dets = curr_dets or []

    if len(curr_dets) == 0 and len(prev_dets) == 0:
        return 1.0

    if len(curr_dets) == 0 and len(prev_dets) > 0:
        return 0.0

    used_prev = [False] * len(prev_dets)
    matched = 0

    for (l2, t2, w2, h2, conf2, cls2) in curr_dets:
        best_iou = 0.0
        best_j = -1

        for j, (l1, t1, w1, h1, conf1, cls1) in enumerate(prev_dets):
            if used_prev[j]:
                continue
            if cls1 != cls2:
                continue

            iou = _bbox_iou_xywh((l1, t1, w1, h1), (l2, t2, w2, h2))
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            used_prev[best_j] = True
            matched += 1

    return matched / float(max(len(curr_dets), 1))


# ============================================================
# 4.1 便捷函数：两个帧之间的 PTC_iou（输入 images）
# ============================================================

def compute_ptc_iou_pair_from_images(
    prev_img_path: str,
    curr_img_path: str,
    model_name: str,
    conf_threshold: float = 0.3,
    iou_thr: float = 0.5,
    ignored_regions_xywh: Optional[List[Tuple[float, float, float, float]]] = None,
) -> float:
    """
    Convenience wrapper:
    输入两张图片路径，内部运行 detector 得到 dets，然后计算 PTC_iou_pair。

    返回:
      float in [0, 1]
    """
    predict_fn = make_predict_fn_dets_only(
        model_name=model_name,
        conf_threshold=conf_threshold,
        ignored_regions_xywh=ignored_regions_xywh,
    )

    prev_dets = predict_fn(prev_img_path)
    curr_dets = predict_fn(curr_img_path)

    return compute_ptc_iou_pair(prev_dets, curr_dets, iou_thr=iou_thr)


# ============================================================
# 5) 序列版本（由原子函数堆叠）+ 平滑
# ============================================================

def _smooth_moving_average(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.copy()

    N = x.shape[0]
    y = np.zeros_like(x, dtype=np.float32)
    half = k // 2

    for i in range(N):
        l = max(0, i - half)
        r = min(N, i + half + 1)
        y[i] = np.mean(x[l:r])

    return y


def compute_ptc_iou_series(
    dets_all: List[list],
    iou_thr: float = 0.5,
    smooth_window: int = 3,
) -> np.ndarray:
    """
    序列版本：
      PTC_series[t] = PTC_pair(frame_{t-1}, frame_t)
    """
    T = len(dets_all)
    ptc = np.ones(T, dtype=np.float32)

    for t in range(T):
        if t == 0:
            ptc[t] = 1.0
            continue
        ptc[t] = compute_ptc_iou_pair(dets_all[t - 1], dets_all[t], iou_thr=iou_thr)

    ptc = _smooth_moving_average(ptc, k=smooth_window)
    return np.clip(ptc, 0.0, 1.0)


# ============================================================
# 6) 输出 CSV
# ============================================================

def save_ptc_csv(csv_path: str, frame_paths: List[str], ptc_iou: np.ndarray):
    """
    保存 CSV：
      columns: frame_index, image_name, image_path, ptc_iou
    """
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "image_name", "image_path", "ptc_iou"])
        for i, p in enumerate(frame_paths):
            writer.writerow([i, os.path.basename(p), p, float(ptc_iou[i])])


# ============================================================
# 7) 运行封装：跑一个 sequence（给 CLI / 研究用）
# ============================================================

def run_ptc_iou_on_sequence(
    img_dir: str,
    seq_name: str,
    model_name: str,
    conf: float,
    out_root: str,
    seq_xml_root: str = "",
    iou_thr: float = 0.5,
    smooth: int = 3,
    max_frames: int = -1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    跑单个 sequence + 单个 model/conf 组合。保存 npy/csv，并返回结果字典。
    """
    paths = load_frame_paths_from_dir(img_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]

    ignored_regions = []
    if seq_xml_root:
        seq_xml_path = os.path.join(seq_xml_root, f"{seq_name}.xml")
        ignored_regions = load_ignored_regions_from_sequence_xml(seq_xml_path)
        if verbose:
            if len(ignored_regions) == 0:
                print(f"[WARN] {seq_name}: no ignored_region found (or xml missing): {seq_xml_path}")
            else:
                print(f"[INFO] {seq_name}: ignored_regions={len(ignored_regions)} (from {seq_xml_path})")
    else:
        if verbose:
            print(f"[INFO] {seq_name}: seq_xml_root not provided -> ignored filter disabled.")

    out_dir = os.path.join(out_root, seq_name, model_name, f"conf{conf}")
    os.makedirs(out_dir, exist_ok=True)

    predict_fn = make_predict_fn_dets_only(
        model_name=model_name,
        conf_threshold=conf,
        ignored_regions_xywh=ignored_regions if ignored_regions else None,
    )

    if verbose:
        print(" → Running detector on frames (collect dets) ...")

    dets_all = []
    for i, img_path in enumerate(paths):
        dets_all.append(predict_fn(img_path))
        if verbose and (i + 1) % 50 == 0:
            print(f"   processed {i + 1}/{len(paths)} frames")

    if verbose:
        print(" → Computing PTC_iou series ...")

    ptc_iou = compute_ptc_iou_series(
        dets_all=dets_all,
        iou_thr=iou_thr,
        smooth_window=smooth
    )

    stats = {
        "min": float(ptc_iou.min()) if ptc_iou.size else 0.0,
        "max": float(ptc_iou.max()) if ptc_iou.size else 0.0,
        "mean": float(ptc_iou.mean()) if ptc_iou.size else 0.0,
        "std": float(ptc_iou.std()) if ptc_iou.size else 0.0,
    }

    base = f"{seq_name}_{model_name}_conf{conf}"
    suffix = "ptc_iou_only"

    out_npy = os.path.join(out_dir, f"{base}_PTC_iou_{suffix}.npy")
    np.save(out_npy, ptc_iou)

    out_csv = os.path.join(out_dir, f"{base}_PTC_iou_{suffix}.csv")
    save_ptc_csv(out_csv, paths, ptc_iou)

    if verbose:
        print("PTC_iou: min,max,mean,std =",
              stats["min"], stats["max"], stats["mean"], stats["std"])
        print(f" ✓ Saved NPY: {out_npy}")
        print(f" ✓ Saved CSV: {out_csv}\n")

    return {
        "seq_name": seq_name,
        "model_name": model_name,
        "conf": conf,
        "num_frames": len(paths),
        "out_dir": out_dir,
        "out_npy": out_npy,
        "out_csv": out_csv,
        "ptc_iou": ptc_iou,
        "stats": stats,
    }


# ============================================================
# 8) 运行封装：跑整个 img_root
# ============================================================

def run_ptc_iou_on_img_root(
    img_root: str,
    out_root: str,
    models: List[str],
    confs: List[float],
    seq_xml_root: str = "",
    iou_thr: float = 0.5,
    smooth: int = 3,
    max_frames: int = -1,
    seqs: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    跑 img_root 下所有序列（或指定 seqs），对每个 model/conf 组合计算 PTC-iou，并保存输出。
    """
    os.makedirs(out_root, exist_ok=True)

    if seqs is None:
        seq_list = sorted([
            d for d in os.listdir(img_root)
            if os.path.isdir(os.path.join(img_root, d))
        ])
    else:
        seq_list = seqs

    if verbose:
        print(f"[INFO] Found {len(seq_list)} sequences under: {img_root}")
        print("       First few:", seq_list[:5])

    results: List[Dict[str, Any]] = []

    for seq_name in seq_list:
        img_dir = os.path.join(img_root, seq_name)

        paths = load_frame_paths_from_dir(img_dir)
        if len(paths) == 0:
            if verbose:
                print(f"[SKIP] No images found in: {img_dir}")
            continue

        if verbose:
            print("\n" + "#" * 80)
            print(f"[SEQ] Processing {seq_name}  (frames={len(paths) if max_frames <= 0 else min(len(paths), max_frames)})")
            print("#" * 80)

        total_tasks = len(models) * len(confs)
        task_id = 1

        for model_name in models:
            for conf in confs:
                if verbose:
                    print("=" * 70)
                    print(f"[{task_id}/{total_tasks}] Running PTC_iou (no GT)")
                    print(f"Sequence    : {seq_name}")
                    print(f"Model       : {model_name}")
                    print(f"Conf thresh : {conf}")
                    print("=" * 70)
                task_id += 1

                res = run_ptc_iou_on_sequence(
                    img_dir=img_dir,
                    seq_name=seq_name,
                    model_name=model_name,
                    conf=conf,
                    out_root=out_root,
                    seq_xml_root=seq_xml_root,
                    iou_thr=iou_thr,
                    smooth=smooth,
                    max_frames=max_frames,
                    verbose=verbose,
                )
                results.append(res)

    if verbose:
        print("All PTC_iou computation finished!")

    return results