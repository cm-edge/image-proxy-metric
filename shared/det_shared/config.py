from pathlib import Path

# Dataset paths
DATA_ROOT = Path(__file__).parent / "data" / "UA-DETRAC"
IMAGES_DIR = DATA_ROOT / "DETRAC-Images"
DETECTIONS_DIR = DATA_ROOT / "DETRAC-Detections"

# Available models
AVAILABLE_MODELS = [
    "fasterrcnn_resnet50_fpn_v2",  # 	280.37 gflops
    "fasterrcnn_mobilenet_v3_large_fpn", # 4.49 gflops
    "retinanet_resnet50_fpn_v2", # 152.24
    "fasterrcnn_mobilenet_v3_large_320_fpn", # 0.72
    "ssd300_vgg16" # 34.86
]

# Default detection parameters
DEFAULT_MODELS = [
    "fasterrcnn_resnet50_fpn_v2", 
    "fasterrcnn_mobilenet_v3_large_fpn", 
    "retinanet_resnet50_fpn_v2", 
    "fasterrcnn_mobilenet_v3_large_320_fpn", 
    "ssd300_vgg16"
    ]  # Can specify multiple models
DEFAULT_CONF = 0.0  # Save all detections (filter by confidence threshold on-the-fly when using)
DEFAULT_BATCH_SIZE = 32  # Batch size for inference (increase for faster processing, decrease if OOM)

# COCO vehicle classes (car, motorcycle, bus, truck)
# COCO class IDs: 3=car, 4=motorcycle, 6=bus, 8=truck
VEHICLE_CLASSES = {3, 4, 6, 8}

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_device():
    """Detect and return the best available device."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
