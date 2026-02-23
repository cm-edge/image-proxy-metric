from ptc_iou_tool.ptc_iou_tool.core import run_ptc_iou_on_img_root

if __name__ == "__main__":

    img_root = r"A:\dataset\DETRAC-Images-Subsets"
    out_root = r"A:\dataset\out"

    models = [
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    ]

    confs = [0.5]

    results = run_ptc_iou_on_img_root(
        img_root=img_root,
        out_root=out_root,
        models=models,
        confs=confs,
        seq_xml_root=r"A:\dataset\DETRAC-Train-Annotations-XML",
        iou_thr=0.5,
        smooth=1,
        max_frames=200,   # 先少跑一点测试
        verbose=True
    )

    print("\nDONE.")