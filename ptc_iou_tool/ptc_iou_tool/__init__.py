"""
PTC-IoU: GT-free time consistency metric for object detection.
"""

from .core import (
    # atomic + image wrapper + series
    compute_ptc_iou_pair,
    compute_ptc_iou_pair_from_images,
    compute_ptc_iou_series,

    # helpers
    load_frame_paths_from_dir,
    load_ignored_regions_from_sequence_xml,
    make_predict_fn_dets_only,
    save_ptc_csv,

    # runners (used by CLI)
    run_ptc_iou_on_sequence,
    run_ptc_iou_on_img_root,
)

__all__ = [
    "compute_ptc_iou_pair",
    "compute_ptc_iou_pair_from_images",
    "compute_ptc_iou_series",
    "load_frame_paths_from_dir",
    "load_ignored_regions_from_sequence_xml",
    "make_predict_fn_dets_only",
    "save_ptc_csv",
    "run_ptc_iou_on_sequence",
    "run_ptc_iou_on_img_root",
]

__version__ = "0.1.0"