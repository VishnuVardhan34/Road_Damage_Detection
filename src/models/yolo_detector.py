"""
YOLO detector wrapper for road damage detection.
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLO detector for road damage detection.
    Supports YOLOv8 and YOLOv11.
    """
    
    CLASS_MAPPING = {
        0: 'Crack',
        1: 'Pothole',
        2: 'Rutting',
        3: 'Patch',
        4: 'Lane_Wear',
        5: 'Manhole'
    }
    
    def __init__(self, model_path: str, device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 use_half: bool = True):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            device: Device to use ('cuda' or 'cpu')
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            use_half: Use FP16 for inference
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_half = use_half
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        if use_half and device == 'cuda':
            self.model.half()
        
        logger.info("YOLO model loaded successfully")
    
    def detect(self, image: np.ndarray, image_size: int = 1024) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Input image (BGR format)
            image_size: Input size for YOLO
            
        Returns:
            Dictionary with detections
        """
        results = self.model(image, imgsz=image_size, conf=self.conf_threshold)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                h, w = image.shape[:2]
                
                for box in boxes:
                    # Get coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Normalize to [0-1]
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.CLASS_MAPPING.get(class_id, 'Unknown'),
                        'bbox': (x_center, y_center, width, height),
                        'confidence': confidence,
                        'bbox_pixels': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        return {
            'image': image,
            'image_size': image_size,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def detect_batch(self, images: List[np.ndarray], 
                    image_size: int = 1024) -> List[Dict]:
        """
        Run inference on batch of images.
        
        Args:
            images: List of input images
            image_size: Input size for YOLO
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(images, imgsz=image_size, conf=self.conf_threshold)
        
        batch_results = []
        for i, result in enumerate(results):
            detections = self._parse_result(result, images[i].shape)
            
            batch_results.append({
                'image': images[i],
                'detections': detections,
                'num_detections': len(detections)
            })
        
        return batch_results
    
    def _parse_result(self, result, image_shape: Tuple) -> List[Dict]:
        """Parse YOLO result into detections."""
        detections = []
        h, w = image_shape[:2]
        
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Normalize
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.CLASS_MAPPING.get(class_id, 'Unknown'),
                    'bbox': (x_center, y_center, width, height),
                    'confidence': confidence,
                    'bbox_pixels': (int(x1), int(y1), int(x2), int(y2))
                })
        
        return detections
    
    def export_onnx(self, output_path: str, image_size: int = 1024):
        """Export model to ONNX format."""
        try:
            self.model.export(format='onnx', imgsz=image_size)
            logger.info(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
    
    def export_torchscript(self, output_path: str, image_size: int = 1024):
        """Export model to TorchScript format."""
        try:
            self.model.export(format='torchscript', imgsz=image_size)
            logger.info(f"Model exported to TorchScript: {output_path}")
        except Exception as e:
            logger.error(f"Error exporting to TorchScript: {e}")
