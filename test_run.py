import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from tta_locu_tool.tta_locu_tool.locu_core import run_locu_on_sequence

res = run_locu_on_sequence(
    seq_name="MVI_20011",
    img_dir=r"A:/dataset/DETRAC-Images-Train/MVI_20011",
    seq_xml_root=r"A:/dataset/DETRAC-Train-Annotations-XML",
    out_root=r"A:/tmp/locu_out",
    model_name="fasterrcnn_mobilenet_v3_large_320_fpn",

    # 为了演示快一点
    aug_types=["gamma"],
    n_runs=10,
    max_frames=20,
)

print("Output dir:", res["out_dir"])
print("Loc-U mean:", float(res["locu"].mean()))