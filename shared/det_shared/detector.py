import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2)

from det_shared.config import COCO_CLASSES, VEHICLE_CLASSES, get_device


class Detector:
    def __init__(self, model_name, conf_threshold):
        self.device = get_device()
        self.conf_threshold = conf_threshold
        
        # Load model based on name
        if "mobilenet" in model_name.lower():
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        elif "retinanet" in model_name.lower():
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights)
        else:  # default: fasterrcnn_resnet50
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()
        
        print(f"Using device: {self.device.upper()}")
        print(f"Model: {model_name}")
    
    def detect(self, image_path):
        """Run detection on a single image, return filtered results."""
        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        # Filter detections
        detections = []
        boxes = predictions['boxes'].cpu()
        labels = predictions['labels'].cpu()
        scores = predictions['scores'].cpu()
        
        for box, label, score in zip(boxes, labels, scores):
            label_id = int(label)
            conf = float(score)
            
            if conf >= self.conf_threshold and label_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.tolist()
                
                # Convert to (left, top, width, height)
                left, top = x1, y1
                width, height = x2 - x1, y2 - y1
                
                class_name = COCO_CLASSES[label_id]
                detections.append((left, top, width, height, conf, class_name))
        
        return detections
    
    def detect_batch(self, image_paths):
        """Run detection on a batch of images, return filtered results for each image."""
        # Load and transform images
        img_tensors = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            img_tensors.append(img_tensor)
        
        # Stack into batch and move to device
        batch = torch.stack(img_tensors).to(self.device)
        
        # Run batch inference
        with torch.no_grad():
            predictions = self.model(batch)
        
        # Filter detections for each image in batch
        batch_detections = []
        for pred in predictions:
            detections = []
            boxes = pred['boxes'].cpu()
            labels = pred['labels'].cpu()
            scores = pred['scores'].cpu()
            
            for box, label, score in zip(boxes, labels, scores):
                label_id = int(label)
                conf = float(score)
                
                if conf >= self.conf_threshold and label_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Convert to (left, top, width, height)
                    left, top = x1, y1
                    width, height = x2 - x1, y2 - y1
                    
                    class_name = COCO_CLASSES[label_id]
                    detections.append((left, top, width, height, conf, class_name))
            
            batch_detections.append(detections)
        
        return batch_detections
