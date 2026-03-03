import argparse
import os
import cv2

from .locu_core import (
    run_locu_on_img_root,
    total_runs,
    compute_locu_for_image,
    make_predict_fn,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("TTA Loc-U tool (GT-free; uses ignored_region only)")

    # ------------------------------
    # Mode selection
    # ------------------------------
    ap.add_argument(
        "--mode",
        choices=["dataset", "single"],
        default="dataset",
        help="dataset = run on img_root; single = run atomic on one image.",
    )

    # ------------------------------
    # Dataset mode arguments
    # ------------------------------
    ap.add_argument("--img_root", help="Root directory containing sequence folders of images.")
    ap.add_argument("--seq_xml_root", help="Root directory containing SEQ_NAME.xml.")
    ap.add_argument("--out_root", help="Output root directory.")
    ap.add_argument("--tmp_root", default="", help="Temporary directory for augmented images.")

    # ------------------------------
    # Single image mode
    # ------------------------------
    ap.add_argument("--image", help="Path to single image (for --mode single).")

    # ------------------------------
    # Model
    # ------------------------------
    ap.add_argument("--models", nargs="+", help="Model names for dataset mode.")
    ap.add_argument("--model", help="Single model name for single-image mode.")
    ap.add_argument("--detector_conf", type=float, default=0.0)

    # ------------------------------
    # TTA
    # ------------------------------
    ap.add_argument("--tta_mode", choices=["grid", "random"], default="grid")
    ap.add_argument(
        "--aug_types",
        nargs="+",
        default=["gamma", "brightness", "contrast"],
        choices=["gamma", "brightness", "contrast"],
    )
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmin", type=float, default=None)
    ap.add_argument("--kmax", type=float, default=None)

    # ------------------------------
    # Loc-U params
    # ------------------------------
    ap.add_argument("--sel_thr", type=float, default=0.22)
    ap.add_argument("--iou_loc", type=float, default=0.22)
    ap.add_argument("--nms_iou", type=float, default=0.45)
    ap.add_argument("--min_cov", type=int, default=9)

    # ------------------------------
    # Controls
    # ------------------------------
    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--max_seqs", type=int, default=-1)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--write_config", action="store_true")
    ap.add_argument("--save_frame_paths", action="store_true")

    return ap


def run_single_image(args):
    if not args.image:
        raise ValueError("--image required for --mode single")
    if not args.model:
        raise ValueError("--model required for --mode single")

    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Failed to load image")

    predict_fn = make_predict_fn(
        model_name=args.model,
        conf_threshold=args.detector_conf,
        ignored_regions=None,
    )

    locu, nc, nk = compute_locu_for_image(
        img=img,
        predict_fn=predict_fn,
        tta_mode=args.tta_mode,
        aug_types=args.aug_types,
        n_runs=args.n_runs,
        seed=args.seed,
        kmin=args.kmin,
        kmax=args.kmax,
        sel_thr=args.sel_thr,
        iou_loc=args.iou_loc,
        nms_iou=args.nms_iou,
        min_cov=args.min_cov,
    )

    print("======================================")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print("--------------------------------------")
    print(f"Loc-U: {locu:.6f}")
    print(f"Centers: {nc}")
    print(f"Kept clusters: {nk}")
    print("======================================\n")


def run_dataset(args, verbose):
    required = ["img_root", "seq_xml_root", "out_root", "models"]
    for r in required:
        if getattr(args, r) is None:
            raise ValueError(f"--{r} is required for --mode dataset")

    tr = total_runs(args.tta_mode, args.aug_types, args.n_runs)
    if args.min_cov > tr:
        raise ValueError(f"--min_cov ({args.min_cov}) cannot be > total_runs ({tr}).")

    if verbose:
        print(f"[INFO] models={args.models}")
        print(f"       total_runs={tr}")
        print("")

    run_locu_on_img_root(
        img_root=args.img_root,
        seq_xml_root=args.seq_xml_root,
        out_root=args.out_root,
        models=args.models,
        detector_conf=args.detector_conf,
        tta_mode=args.tta_mode,
        aug_types=args.aug_types,
        n_runs=args.n_runs,
        seed=args.seed,
        kmin=args.kmin,
        kmax=args.kmax,
        sel_thr=args.sel_thr,
        iou_loc=args.iou_loc,
        nms_iou=args.nms_iou,
        min_cov=args.min_cov,
        tmp_root=args.tmp_root,
        keep_tmp=args.keep_tmp,
        seqs=args.seqs,
        max_seqs=args.max_seqs,
        max_frames=args.max_frames,
        write_config=args.write_config,
        save_frame_paths=args.save_frame_paths,
        verbose=verbose,
    )


def main():
    args = build_parser().parse_args()
    verbose = not args.quiet

    if args.mode == "single":
        run_single_image(args)
    else:
        run_dataset(args, verbose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")