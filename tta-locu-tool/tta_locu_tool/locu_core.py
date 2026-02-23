import os
import glob
import csv
from typing import List, Tuple, Optional, Set, Dict, Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from typing import Callable

from det_shared.detector import Detector


# ============================================================
# 0) Types
# ============================================================
Det = Tuple[float, float, float, float, float, str]   # (l,t,w,h,conf,cls)
BBox = Tuple[float, float, float, float]              # (l,t,w,h)
DetWithRun = Tuple[Det, int]                          # (det, run_id)


# ============================================================
# 1) ignored_region: load + filter by center
# ============================================================
def load_ignored_regions_from_sequence_xml(seq_xml_path: str) -> List[BBox]:
    if (not seq_xml_path) or (not os.path.isfile(seq_xml_path)):
        return []
    tree = ET.parse(seq_xml_path)
    root = tree.getroot()
    ignored: List[BBox] = []
    ign = root.find("ignored_region")
    if ign is None:
        return ignored
    for box in ign.findall("box"):
        l = float(box.get("left", "0"))
        t = float(box.get("top", "0"))
        w = float(box.get("width", "0"))
        h = float(box.get("height", "0"))
        ignored.append((l, t, w, h))
    return ignored


def _xywh_to_xyxy(b: BBox) -> Tuple[float, float, float, float]:
    l, t, w, h = b
    return (l, t, l + w, t + h)


def _center_in_region(det_xyxy, reg_xyxy) -> bool:
    x1, y1, x2, y2 = det_xyxy
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    rx1, ry1, rx2, ry2 = reg_xyxy
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def filter_dets_ignored_center(dets: List[Det], ignored_regions_xywh: List[BBox]) -> List[Det]:
    if not ignored_regions_xywh:
        return dets
    regs = [_xywh_to_xyxy(r) for r in ignored_regions_xywh]
    out: List[Det] = []
    for (l, t, w, h, s, c) in dets:
        dxyxy = _xywh_to_xyxy((l, t, w, h))
        if any(_center_in_region(dxyxy, rxyxy) for rxyxy in regs):
            continue
        out.append((l, t, w, h, s, c))
    return out


# ============================================================
# 2) IO helpers
# ============================================================
def load_image_paths(seq_img_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(seq_img_dir, ext)))
    return sorted(paths)


# ============================================================
# 3) Detector wrapper (path-in, dets-out + ignored filter)
# ============================================================
def make_predict_fn(model_name: str, conf_threshold: float, ignored_regions: Optional[List[BBox]]):
    det = Detector(model_name, conf_threshold)

    def predict(img_path: str) -> List[Det]:
        dets0 = det.detect(img_path)
        dets: List[Det] = []
        for l, t, w, h, s, c in dets0:
            dets.append((float(l), float(t), float(w), float(h), float(s), str(c)))
        if ignored_regions:
            dets = filter_dets_ignored_center(dets, ignored_regions)
        return dets

    return predict


# ============================================================
# 4) TTA (paper-style + random sampling)
# ============================================================
def _apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    img = np.clip(img, 0, 1)
    img = np.power(img, gamma)
    return (img * 255.0).astype(np.uint8)


def _apply_brightness(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    out = img_bgr.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_contrast(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    f = float(factor)
    out = (img_bgr.astype(np.float32) - 128.0) * f + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def default_k_range(aug_type: str) -> Tuple[float, float]:
    # match your original defaults
    if aug_type == "gamma":
        return 0.4, 2.0
    return 0.3, 2.0


def generate_paper_tta_images(
    img_bgr: np.ndarray,
    aug_type: str,
    n_runs: int,
    kmin: float,
    kmax: float,
) -> List[Tuple[str, np.ndarray]]:
    """
    Deterministic sweep: N evenly spaced k in [kmin,kmax].
    Returns list of (tag, image_bgr).
    """
    assert n_runs >= 1
    if n_runs == 1:
        ks = [kmin]
    else:
        ks = [kmin + i * (kmax - kmin) / (n_runs - 1) for i in range(n_runs)]

    out: List[Tuple[str, np.ndarray]] = []
    for k in ks:
        if aug_type == "gamma":
            im = _apply_gamma(img_bgr, k)
        elif aug_type == "brightness":
            im = _apply_brightness(img_bgr, k)
        elif aug_type == "contrast":
            im = _apply_contrast(img_bgr, k)
        else:
            raise ValueError(f"Unknown aug_type: {aug_type}")
        out.append((f"{aug_type}_{k:.3f}", im))
    return out


def generate_random_tta_images(
    img_bgr: np.ndarray,
    aug_types: List[str],
    total_runs: int,
    seed: int,
    kmin_global: Optional[float],
    kmax_global: Optional[float],
) -> List[Tuple[str, np.ndarray]]:
    """
    Random sampling: total_runs times, each run randomly picks aug_type and k~Uniform(kmin,kmax).
    If kmin_global/kmax_global provided -> use same range for all aug types, else per-type default range.
    Returns list of (tag, image_bgr).
    """
    rng = np.random.default_rng(int(seed))

    out: List[Tuple[str, np.ndarray]] = []
    for _ in range(int(total_runs)):
        aug = str(rng.choice(aug_types))

        if (kmin_global is not None) and (kmax_global is not None):
            kmin, kmax = float(kmin_global), float(kmax_global)
        else:
            kmin, kmax = default_k_range(aug)

        k = float(rng.uniform(kmin, kmax))

        if aug == "gamma":
            im = _apply_gamma(img_bgr, k)
        elif aug == "brightness":
            im = _apply_brightness(img_bgr, k)
        elif aug == "contrast":
            im = _apply_contrast(img_bgr, k)
        else:
            raise ValueError(aug)

        out.append((f"{aug}_rand{k:.3f}", im))
    return out


# ============================================================
# 5) Loc-U building blocks
# ============================================================
def iou_xywh(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def nms_xywh(dets: List[Det], iou_thr: float = 0.5, class_aware: bool = True) -> List[Det]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
    kept: List[Det] = []
    for d in dets_sorted:
        l, t, w, h, s, c = d
        ok = True
        for kd in kept:
            kl, kt, kw, kh, ks, kc = kd
            if class_aware and kc != c:
                continue
            if iou_xywh((l, t, w, h), (kl, kt, kw, kh)) > iou_thr:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def _xywh_to_norm_vec(l, t, w, h, W, H) -> np.ndarray:
    cx = (l + 0.5 * w) / float(W)
    cy = (t + 0.5 * h) / float(H)
    nw = w / float(W)
    nh = h / float(H)
    return np.array([cx, cy, nw, nh], dtype=np.float32)


def _gaussian_entropy(cov: np.ndarray, eps: float = 1e-8) -> float:
    d = cov.shape[0]
    cov = cov.astype(np.float64) + np.eye(d, dtype=np.float64) * eps
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = np.log(max(np.linalg.det(cov), eps))
    return float(0.5 * logdet + 0.5 * d * (1.0 + np.log(2.0 * np.pi)))


def loc_u_from_clusters(clusters_loc: List[List[Det]], img_wh: Tuple[int, int]) -> float:
    W, H = img_wh
    ents: List[float] = []
    for members in clusters_loc:
        vecs = [_xywh_to_norm_vec(l, t, w, h, W, H) for (l, t, w, h, s, c) in members]
        vecs = np.stack(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        X = vecs - mu
        cov = (X.T @ X) / float(max(vecs.shape[0], 1))
        ents.append(_gaussian_entropy(cov))
    return float(np.mean(ents)) if ents else 0.0


def cluster_around_centers_with_runs(
    centers: List[Det],
    preds_wr: List[DetWithRun],
    iou_thr: float,
    same_label: bool = False,
) -> List[Tuple[List[Det], Set[int]]]:
    clusters: List[Tuple[List[Det], Set[int]]] = []
    for c in centers:
        cl, ct, cw, ch, cs, cc = c
        members: List[Det] = []
        run_ids: Set[int] = set()

        for (p, rid) in preds_wr:
            pl, pt, pw, ph, ps, pc = p
            if same_label and (pc != cc):
                continue
            if iou_xywh((cl, ct, cw, ch), (pl, pt, pw, ph)) >= iou_thr:
                members.append(p)
                run_ids.add(rid)

        if not members:
            continue
        clusters.append((members, run_ids))
    return clusters


def compute_loc_u_for_frame_fixed(
    dets_each_run: List[List[Det]],   # length = total_runs
    img_wh: Tuple[int, int],
    sel_thr: float,
    nms_iou: float,
    iou_loc: float,
    min_cov: int,
) -> Tuple[float, int, int]:
    """
    Returns:
      loc_u, num_centers, num_kept_clusters
    """
    P_wr: List[DetWithRun] = []
    for rid, dets in enumerate(dets_each_run):
        for d in dets:
            if float(d[4]) >= sel_thr:
                P_wr.append((d, rid))

    if not P_wr:
        return 0.0, 0, 0

    P_only: List[Det] = [d for (d, _) in P_wr]
    centers = nms_xywh(P_only, iou_thr=nms_iou, class_aware=True)
    num_centers = len(centers)
    if num_centers == 0:
        return 0.0, 0, 0

    clusters_wr = cluster_around_centers_with_runs(
        centers=centers, preds_wr=P_wr, iou_thr=iou_loc, same_label=False
    )

    kept_clusters: List[List[Det]] = []
    for members, run_ids in clusters_wr:
        if len(run_ids) >= int(min_cov):
            kept_clusters.append(members)

    if not kept_clusters:
        return 0.0, num_centers, 0

    loc_u = loc_u_from_clusters(kept_clusters, img_wh=img_wh)
    return float(loc_u), int(num_centers), int(len(kept_clusters))


# ============================================================
# 6) CSV writer
# ============================================================
def save_frame_csv(
    csv_path: str,
    frame_paths: List[str],
    locu: np.ndarray,
    num_centers: np.ndarray,
    num_kept: np.ndarray,
):
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "image_name", "image_path", "loc_u", "num_centers", "num_kept_clusters"])
        for i, p in enumerate(frame_paths):
            w.writerow([i, os.path.basename(p), p, float(locu[i]), int(num_centers[i]), int(num_kept[i])])


# ============================================================
# 7) Utilities for runner
# ============================================================
def total_runs(tta_mode: str, aug_types: List[str], n_runs: int) -> int:
    if tta_mode == "grid":
        return len(aug_types) * int(n_runs)
    if tta_mode == "random":
        return int(n_runs)
    raise ValueError(f"Unknown tta_mode: {tta_mode}")


# ============================================================
# 8) Runner: one sequence + one model
# ============================================================
def run_locu_on_sequence(
    seq_name: str,
    img_dir: str,
    seq_xml_root: str,
    out_root: str,
    model_name: str,
    detector_conf: float = 0.0,
    tta_mode: str = "grid",
    aug_types: Optional[List[str]] = None,
    n_runs: int = 10,
    seed: int = 0,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    sel_thr: float = 0.22,
    iou_loc: float = 0.22,
    nms_iou: float = 0.45,
    min_cov: int = 9,
    tmp_root: str = "",
    keep_tmp: bool = False,
    max_frames: int = -1,
    write_config: bool = False,
    save_frame_paths: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Loc-U for ONE sequence and ONE model. Saves npy/csv and returns a summary dict.
    Behavior matches your original script (including saving augmented images to disk).
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if aug_types is None:
        aug_types = ["gamma", "brightness", "contrast"]

    img_paths = load_image_paths(img_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images under: {img_dir}")

    if max_frames and max_frames > 0:
        img_paths = img_paths[: max_frames]

    os.makedirs(out_root, exist_ok=True)
    tmp_root_eff = tmp_root.strip() or os.path.join(out_root, "_tmp_tta")
    os.makedirs(tmp_root_eff, exist_ok=True)

    tot_runs = total_runs(tta_mode=tta_mode, aug_types=aug_types, n_runs=n_runs)
    if min_cov > tot_runs:
        raise ValueError(f"min_cov ({min_cov}) cannot be > total_runs ({tot_runs}).")

    # ignored regions
    seq_xml_path = os.path.join(seq_xml_root, f"{seq_name}.xml")
    ignored = load_ignored_regions_from_sequence_xml(seq_xml_path)
    ignored_opt = ignored if ignored else None

    if verbose:
        print("\n" + "#" * 80)
        print(f"[SEQ] {seq_name}  frames={len(img_paths)}  ignored_regions={len(ignored)}")
        print("#" * 80)
        print(f"[MODEL] {model_name}")
        print(f"tta_mode={tta_mode} aug_types={aug_types} n_runs={n_runs} => total_runs={tot_runs}")
        print(f"combo: sel_thr={sel_thr} iou_loc={iou_loc} nms_iou={nms_iou} min_cov={min_cov}/{tot_runs}")
        print(f"tmp_root={tmp_root_eff} keep_tmp={keep_tmp}")
        print("")

    predict_fn = make_predict_fn(model_name, conf_threshold=detector_conf, ignored_regions=ignored_opt)

    # output dir naming includes mode + totals to avoid confusion
    if tta_mode == "grid":
        mode_str = f"grid_{'+'.join(aug_types)}_runs{n_runs}peraug_total{tot_runs}"
    else:
        mode_str = f"random_{'+'.join(aug_types)}_runs{n_runs}total_seed{seed}"

    cfg_str = (
        f"tta_{mode_str}_"
        f"sel{sel_thr}_ioloc{iou_loc}_nms{nms_iou}_cov{min_cov}of{tot_runs}"
    )
    out_dir = os.path.join(out_root, seq_name, model_name, cfg_str)
    os.makedirs(out_dir, exist_ok=True)

    if write_config:
        with open(os.path.join(out_dir, "config.txt"), "w", encoding="utf-8") as f:
            f.write(f"SEQ={seq_name}\nMODEL={model_name}\n")
            f.write(f"IMG_DIR={img_dir}\nSEQ_XML_ROOT={seq_xml_root}\n")
            f.write(f"DETECTOR_CONF={detector_conf}\n")
            f.write(f"TTA_MODE={tta_mode}\nAUG_TYPES={aug_types}\n")
            f.write(f"N_RUNS={n_runs}\nTOTAL_RUNS={tot_runs}\n")
            f.write(f"SEED_BASE={seed}\n")
            f.write(f"GLOBAL_KMIN={kmin}\nGLOBAL_KMAX={kmax}\n")
            f.write(f"SEL_THR={sel_thr}\nIOU_LOC={iou_loc}\nNMS_IOU={nms_iou}\n")
            f.write(f"MIN_COV={min_cov}\nKEEP_TMP={keep_tmp}\n")

    T = len(img_paths)
    locu = np.zeros(T, dtype=np.float32)
    num_centers = np.zeros(T, dtype=np.int32)
    num_kept = np.zeros(T, dtype=np.int32)

    safe_name = f"{seq_name}_{model_name}".replace(":", "_").replace("\\", "_").replace("/", "_")
    tmp_dir = os.path.join(tmp_root_eff, safe_name)
    os.makedirs(tmp_dir, exist_ok=True)

    for t, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            locu[t] = 0.0
            continue
        H, W = img.shape[:2]

        dets_each_run: List[List[Det]] = []
        tmp_paths_this_frame: List[str] = []

        if tta_mode == "grid":
            run_id = 0
            for aug in aug_types:
                if (kmin is not None) and (kmax is not None):
                    kmin_eff, kmax_eff = float(kmin), float(kmax)
                else:
                    kmin_eff, kmax_eff = default_k_range(aug)

                tta_imgs = generate_paper_tta_images(
                    img_bgr=img,
                    aug_type=aug,
                    n_runs=int(n_runs),
                    kmin=kmin_eff,
                    kmax=kmax_eff,
                )

                for (tag, im) in tta_imgs:
                    tmp_path = os.path.join(tmp_dir, f"f{t:06d}_rid{run_id:03d}_{tag}.jpg")
                    cv2.imwrite(tmp_path, im)
                    tmp_paths_this_frame.append(tmp_path)
                    dets_each_run.append(predict_fn(tmp_path))
                    run_id += 1
        else:
            # random mode: total runs = n_runs
            tta_imgs = generate_random_tta_images(
                img_bgr=img,
                aug_types=list(aug_types),
                total_runs=int(n_runs),
                seed=int(seed) + int(t),  # per-frame reproducible seed
                kmin_global=kmin,
                kmax_global=kmax,
            )

            for r, (tag, im) in enumerate(tta_imgs):
                tmp_path = os.path.join(tmp_dir, f"f{t:06d}_r{r:02d}_{tag}.jpg")
                cv2.imwrite(tmp_path, im)
                tmp_paths_this_frame.append(tmp_path)
                dets_each_run.append(predict_fn(tmp_path))

        # cleanup temp
        if not keep_tmp:
            for p in tmp_paths_this_frame:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        val, ncent, nkeep = compute_loc_u_for_frame_fixed(
            dets_each_run=dets_each_run,
            img_wh=(W, H),
            sel_thr=sel_thr,
            nms_iou=nms_iou,
            iou_loc=iou_loc,
            min_cov=min_cov,
        )
        locu[t] = float(val)
        num_centers[t] = int(ncent)
        num_kept[t] = int(nkeep)

        if verbose and (t + 1) % 50 == 0:
            print(f"  processed {t+1}/{T} | loc_u mean={float(np.mean(locu[:t+1])):.4f}")

    # remove tmp dir if clean
    if not keep_tmp:
        try:
            leftover = glob.glob(os.path.join(tmp_dir, "*.jpg"))
            for fpath in leftover:
                os.remove(fpath)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    # save outputs
    npy_locu = os.path.join(out_dir, f"{seq_name}_locu.npy")
    npy_centers = os.path.join(out_dir, f"{seq_name}_num_centers.npy")
    npy_kept = os.path.join(out_dir, f"{seq_name}_num_kept.npy")
    np.save(npy_locu, locu)
    np.save(npy_centers, num_centers)
    np.save(npy_kept, num_kept)

    csv_path = os.path.join(out_dir, f"{seq_name}_locu.csv")
    save_frame_csv(
        csv_path=csv_path,
        frame_paths=img_paths,
        locu=locu,
        num_centers=num_centers,
        num_kept=num_kept,
    )

    if save_frame_paths:
        with open(os.path.join(out_dir, f"{seq_name}_frame_paths.txt"), "w", encoding="utf-8") as f:
            for p in img_paths:
                f.write(p + "\n")

    if verbose:
        print(f"[SAVE] {seq_name}: loc_u min/max/mean = {float(locu.min()):.4f} / {float(locu.max()):.4f} / {float(locu.mean()):.4f}")
        print(f"       -> {out_dir}")

    return {
        "seq_name": seq_name,
        "model_name": model_name,
        "out_dir": out_dir,
        "locu": locu,
        "num_centers": num_centers,
        "num_kept": num_kept,
        "csv_path": csv_path,
        "npy_locu": npy_locu,
        "npy_num_centers": npy_centers,
        "npy_num_kept": npy_kept,
    }


# ============================================================
# 9) Runner: entire img_root (multiple sequences, multiple models)
# ============================================================
def run_locu_on_img_root(
    img_root: str,
    seq_xml_root: str,
    out_root: str,
    models: List[str],
    detector_conf: float = 0.0,
    tta_mode: str = "grid",
    aug_types: Optional[List[str]] = None,
    n_runs: int = 10,
    seed: int = 0,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    sel_thr: float = 0.22,
    iou_loc: float = 0.22,
    nms_iou: float = 0.45,
    min_cov: int = 9,
    tmp_root: str = "",
    keep_tmp: bool = False,
    seqs: Optional[List[str]] = None,
    max_seqs: int = -1,
    max_frames: int = -1,
    write_config: bool = False,
    save_frame_paths: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if aug_types is None:
        aug_types = ["gamma", "brightness", "contrast"]

    # sequence list
    if seqs is None:
        seq_list = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
    else:
        seq_list = list(seqs)

    if max_seqs and max_seqs > 0:
        seq_list = seq_list[: max_seqs]

    if verbose:
        print(f"[INFO] Found {len(seq_list)} sequences under: {img_root}")
        print(f"       models={models}")
        print("")

    results: List[Dict[str, Any]] = []
    for seq in seq_list:
        img_dir = os.path.join(img_root, seq)
        # skip empty
        if not load_image_paths(img_dir):
            if verbose:
                print(f"[SKIP] {seq}: no images")
            continue

        for model_name in models:
            res = run_locu_on_sequence(
                seq_name=seq,
                img_dir=img_dir,
                seq_xml_root=seq_xml_root,
                out_root=out_root,
                model_name=model_name,
                detector_conf=detector_conf,
                tta_mode=tta_mode,
                aug_types=aug_types,
                n_runs=n_runs,
                seed=seed,
                kmin=kmin,
                kmax=kmax,
                sel_thr=sel_thr,
                iou_loc=iou_loc,
                nms_iou=nms_iou,
                min_cov=min_cov,
                tmp_root=tmp_root,
                keep_tmp=keep_tmp,
                max_frames=max_frames,
                write_config=write_config,
                save_frame_paths=save_frame_paths,
                verbose=verbose,
            )
            results.append(res)

    if verbose:
        print("\nAll TTA loc_u computation finished!")

    return results

# ============================================================
# 10) Atomic single-image Loc-U
# ============================================================

def compute_locu_for_image(
    img: np.ndarray,
    predict_fn: Callable[[np.ndarray], List[Det]],
    tta_mode: str = "grid",
    aug_types: Optional[List[str]] = None,
    n_runs: int = 10,
    seed: int = 0,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    sel_thr: float = 0.22,
    iou_loc: float = 0.22,
    nms_iou: float = 0.45,
    min_cov: int = 9,
) -> Tuple[float, int, int]:

    if aug_types is None:
        aug_types = ["gamma", "brightness", "contrast"]

    H, W = img.shape[:2]
    dets_each_run: List[List[Det]] = []

    if tta_mode == "grid":
        for aug in aug_types:

            if (kmin is not None) and (kmax is not None):
                kmin_eff, kmax_eff = float(kmin), float(kmax)
            else:
                kmin_eff, kmax_eff = default_k_range(aug)

            tta_imgs = generate_paper_tta_images(
                img_bgr=img,
                aug_type=aug,
                n_runs=int(n_runs),
                kmin=kmin_eff,
                kmax=kmax_eff,
            )

            for _, im in tta_imgs:
                dets_each_run.append(predict_fn(im))

    elif tta_mode == "random":

        tta_imgs = generate_random_tta_images(
            img_bgr=img,
            aug_types=list(aug_types),
            total_runs=int(n_runs),
            seed=int(seed),
            kmin_global=kmin,
            kmax_global=kmax,
        )

        for _, im in tta_imgs:
            dets_each_run.append(predict_fn(im))

    else:
        raise ValueError(f"Unknown tta_mode: {tta_mode}")

    return compute_loc_u_for_frame_fixed(
        dets_each_run=dets_each_run,
        img_wh=(W, H),
        sel_thr=sel_thr,
        nms_iou=nms_iou,
        iou_loc=iou_loc,
        min_cov=min_cov,
    )