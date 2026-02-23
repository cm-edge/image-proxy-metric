import os
import glob
import csv
import argparse
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET

import numpy as np

from detector import Detector


# ===========================
# 0) ignored_region: 读取 + 过滤
# ===========================
def load_ignored_regions_from_sequence_xml(seq_xml_path: str) -> List[Tuple[float, float, float, float]]:
    """
    读取 DETRAC 序列级 XML 里的 ignored_region，返回 List[(left, top, width, height)]
    如果文件不存在/没有该字段 -> 返回 []
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


def _xywh_to_xyxy(box_xywh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box_xywh
    return (x, y, x + w, y + h)


def _box_center_in_region(det_xyxy, region_xyxy) -> bool:
    x1, y1, x2, y2 = det_xyxy
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    rx1, ry1, rx2, ry2 = region_xyxy
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def filter_detections_by_ignored_regions_center(
    detections: List[Tuple[float, float, float, float, float, str]],
    ignored_regions_xywh: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float, float, str]]:
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


# ===========================
# 1) 读帧工具（只拿路径）
# ===========================
def load_frame_paths_from_dir(dir_path: str) -> List[str]:
    """
    从文件夹读取所有 .jpg / .png / .jpeg，按文件名排序。只返回 paths（不读图）
    """
    exts = ("*.jpg", "*.png", "*.jpeg")
    frame_paths = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(dir_path, ext)))
    frame_paths = sorted(frame_paths)
    return frame_paths


# ===========================
# 2) 预测：只返回 dets（并可选过滤 ignored）
# ===========================
def make_predict_fn_dets_only(
    model_name: str,
    conf_threshold: float = 0.3,
    ignored_regions_xywh: Optional[List[Tuple[float, float, float, float]]] = None,
):
    detector = Detector(model_name, conf_threshold)

    def predict_fn(img_path: str) -> list:
        """
        输入: 图片路径
        输出: dets_out = List[(l, t, w, h, conf, class_name)]
        """
        detections = detector.detect(img_path)

        dets_out = []
        for left, top, width, height, conf, class_name in detections:
            dets_out.append(
                (float(left), float(top), float(width), float(height), float(conf), str(class_name))
            )

        # ignored_region 过滤（中心点规则）
        if ignored_regions_xywh:
            dets_out = filter_detections_by_ignored_regions_center(dets_out, ignored_regions_xywh)

        return dets_out

    return predict_fn


# ===========================
# 3) IoU 工具 + PTC_iou 计算
# ===========================
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
    dets_all: 每帧的 dets 列表
      det 元素: (left, top, width, height, conf, class_name)

    返回:
      PTC_iou: shape (T,), 每帧目标级时间一致性（越大越一致）
    """
    T = len(dets_all)
    PTC_iou = np.ones(T, dtype=np.float32)

    for t in range(T):
        if t == 0:
            PTC_iou[t] = 1.0
            continue

        prev_dets = dets_all[t - 1] or []
        curr_dets = dets_all[t] or []

        if len(curr_dets) == 0 and len(prev_dets) == 0:
            PTC_iou[t] = 1.0
            continue
        if len(curr_dets) == 0 and len(prev_dets) > 0:
            PTC_iou[t] = 0.0
            continue

        used_prev = [False] * len(prev_dets)
        matched = 0

        # 对当前帧每个 det，在上一帧里找“同类且 IoU 最大”的 det
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

        PTC_iou[t] = matched / float(max(len(curr_dets), 1))

    # 平滑一下，弱化单帧抖动
    PTC_iou = _smooth_moving_average(PTC_iou, k=smooth_window)
    PTC_iou = np.clip(PTC_iou, 0.0, 1.0)
    return PTC_iou


# ===========================
# 4) 输出 CSV：每帧/每图的 ptc_iou
# ===========================
def save_ptc_csv(csv_path: str, frame_paths: List[str], ptc_iou: np.ndarray):
    """
    保存 CSV：
      columns: frame_index, image_name, image_path, ptc_iou
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "image_name", "image_path", "ptc_iou"])
        for i, p in enumerate(frame_paths):
            writer.writerow([i, os.path.basename(p), p, float(ptc_iou[i])])


# ===========================
# 5) CLI
# ===========================
def parse_args():
    p = argparse.ArgumentParser("PTC_iou tool (no GT required)")

    p.add_argument("--img_root", required=True,
                   help="Root dir containing sequence folders (each folder contains images).")

    # ignored_region XML 是可选的（不传就不使用 ignored filter）
    p.add_argument("--seq_xml_root", default="",
                   help="Optional. Root dir containing sequence XMLs (ignored_region). If empty, ignored filter disabled.")

    p.add_argument("--out_root", required=True,
                   help="Output root dir. Will save .npy and .csv under this folder.")

    p.add_argument("--models", nargs="+", required=True,
                   help="Model names passed to Detector(model_name, conf).")
    p.add_argument("--confs", nargs="+", type=float, required=True,
                   help="Confidence thresholds, e.g. 0.5 0.3")

    p.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for matching across frames.")
    p.add_argument("--smooth", type=int, default=3, help="Smoothing window. Use 1 to disable.")
    p.add_argument("--max_frames", type=int, default=-1, help="If >0, only process first N frames (debug).")

    p.add_argument("--seqs", nargs="*", default=None,
                   help="Optional. Only run these sequences (folder names). If omitted, run all under img_root.")

    return p.parse_args()


def main():
    args = parse_args()

    IMG_ROOT = args.img_root
    OUT_ROOT = args.out_root
    SEQ_XML_ROOT = args.seq_xml_root

    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.seqs is None:
        seq_list = sorted([
            d for d in os.listdir(IMG_ROOT)
            if os.path.isdir(os.path.join(IMG_ROOT, d))
        ])
    else:
        seq_list = args.seqs

    print(f"[INFO] Found {len(seq_list)} sequences under: {IMG_ROOT}")
    print("       First few:", seq_list[:5])

    for seq_name in seq_list:
        img_dir = os.path.join(IMG_ROOT, seq_name)

        paths = load_frame_paths_from_dir(img_dir)
        if len(paths) == 0:
            print(f"[SKIP] No images found in: {img_dir}")
            continue

        if args.max_frames is not None and args.max_frames > 0:
            paths = paths[:args.max_frames]

        # ignored_region：可选
        ignored_regions = []
        if SEQ_XML_ROOT:
            seq_xml_path = os.path.join(SEQ_XML_ROOT, f"{seq_name}.xml")
            ignored_regions = load_ignored_regions_from_sequence_xml(seq_xml_path)
            if len(ignored_regions) == 0:
                print(f"[WARN] {seq_name}: no ignored_region found (or xml missing): {seq_xml_path}")
            else:
                print(f"[INFO] {seq_name}: ignored_regions={len(ignored_regions)} (from {seq_xml_path})")
        else:
            print(f"[INFO] {seq_name}: seq_xml_root not provided -> ignored filter disabled.")

        print("\n" + "#" * 80)
        print(f"[SEQ] Processing {seq_name}  (frames={len(paths)})")
        print("#" * 80)

        total_tasks = len(args.models) * len(args.confs)
        task_id = 1

        for model_name in args.models:
            for conf in args.confs:
                print("=" * 70)
                print(f"[{task_id}/{total_tasks}] Running PTC_iou (no GT)")
                print(f"Sequence    : {seq_name}")
                print(f"Model       : {model_name}")
                print(f"Conf thresh : {conf}")
                print("=" * 70)
                task_id += 1

                out_dir = os.path.join(OUT_ROOT, seq_name, model_name, f"conf{conf}")
                os.makedirs(out_dir, exist_ok=True)

                predict_fn = make_predict_fn_dets_only(
                    model_name=model_name,
                    conf_threshold=conf,
                    ignored_regions_xywh=ignored_regions if ignored_regions else None
                )

                print(" → Running detector on frames (collect dets) ...")
                dets_all = []
                for i, img_path in enumerate(paths):
                    dets_all.append(predict_fn(img_path))
                    if (i + 1) % 50 == 0:
                        print(f"   processed {i + 1}/{len(paths)} frames")

                print(" → Computing PTC_iou series ...")
                ptc_iou = compute_ptc_iou_series(
                    dets_all=dets_all,
                    iou_thr=args.iou_thr,
                    smooth_window=args.smooth
                )

                print(
                    "PTC_iou: min,max,mean,std =",
                    float(ptc_iou.min()),
                    float(ptc_iou.max()),
                    float(ptc_iou.mean()),
                    float(ptc_iou.std())
                )

                base = f"{seq_name}_{model_name}_conf{conf}"
                suffix = "ptc_iou_tool"

                # 1) npy
                out_npy = os.path.join(out_dir, f"{base}_PTC_iou_{suffix}.npy")
                np.save(out_npy, ptc_iou)

                # 2) csv（每张图对应分数）
                out_csv = os.path.join(out_dir, f"{base}_PTC_iou_{suffix}.csv")
                save_ptc_csv(out_csv, paths, ptc_iou)

                print(f" ✓ Saved NPY: {out_npy}")
                print(f" ✓ Saved CSV: {out_csv}\n")

    print("All PTC_iou computation finished!")


if __name__ == "__main__":
    # 让 Ctrl+C 中断时更友好（不影响正常运行）
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting gracefully.")
