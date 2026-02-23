import argparse

from .core import run_ptc_iou_on_img_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PTC_iou tool (no GT required)")

    p.add_argument(
        "--img_root",
        required=True,
        help="Root dir containing sequence folders (each folder contains images).",
    )

    # ignored_region XML 是可选的（不传就不使用 ignored filter）
    p.add_argument(
        "--seq_xml_root",
        default="",
        help="Optional. Root dir containing sequence XMLs (ignored_region). If empty, ignored filter disabled.",
    )

    p.add_argument(
        "--out_root",
        required=True,
        help="Output root dir. Will save .npy and .csv under this folder.",
    )

    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names passed to Detector(model_name, conf).",
    )
    p.add_argument(
        "--confs",
        nargs="+",
        type=float,
        required=True,
        help="Confidence thresholds, e.g. 0.5 0.3",
    )

    p.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for matching across frames.")
    p.add_argument("--smooth", type=int, default=3, help="Smoothing window. Use 1 to disable.")
    p.add_argument("--max_frames", type=int, default=-1, help="If >0, only process first N frames (debug).")

    p.add_argument(
        "--seqs",
        nargs="*",
        default=None,
        help="Optional. Only run these sequences (folder names). If omitted, run all under img_root.",
    )

    p.add_argument("--quiet", action="store_true", help="Disable verbose logs.")
    return p


def main():
    args = build_parser().parse_args()

    run_ptc_iou_on_img_root(
        img_root=args.img_root,
        out_root=args.out_root,
        models=args.models,
        confs=args.confs,
        seq_xml_root=args.seq_xml_root,
        iou_thr=args.iou_thr,
        smooth=args.smooth,
        max_frames=args.max_frames,
        seqs=args.seqs,
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting gracefully.")