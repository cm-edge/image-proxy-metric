"""
tta-locu-tool: GT-free TTA-based localization uncertainty (Loc-U)

Public APIs
===========

Atomic:
- compute_loc_u_for_frame_fixed: dets_each_run -> loc_u
- compute_locu_for_image: image -> loc_u (with TTA)

Runners:
- run_locu_on_sequence: run on a single sequence folder (saves npy/csv)
- run_locu_on_img_root: run on img_root with multiple sequences/models (saves npy/csv)

Utilities:
- make_predict_fn
- total_runs

Types:
- Det
- BBox
- DetWithRun
"""

from .locu_core import (
    # ----------------------------
    # Types
    # ----------------------------
    Det,
    BBox,
    DetWithRun,

    # ----------------------------
    # Atomic APIs
    # ----------------------------
    compute_loc_u_for_frame_fixed,
    compute_locu_for_image,

    # ----------------------------
    # Runners
    # ----------------------------
    run_locu_on_sequence,
    run_locu_on_img_root,

    # ----------------------------
    # Utilities
    # ----------------------------
    make_predict_fn,
    total_runs,
)

__all__ = [
    # Types
    "Det",
    "BBox",
    "DetWithRun",

    # Atomic
    "compute_loc_u_for_frame_fixed",
    "compute_locu_for_image",

    # Runners
    "run_locu_on_sequence",
    "run_locu_on_img_root",

    # Utilities
    "make_predict_fn",
    "total_runs",
]