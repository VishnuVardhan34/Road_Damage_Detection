"""
Inference engine for efficient model serving.
"""

import torch
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-performance inference engine."""
    
    def __init__(self, model_path: str, device: str = 'cuda',
                 use_half: bool = True, batch_size: int = 1):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model
            device: Device to use
            use_half: Use FP16
            batch_size: Batch size for inference
        """
        from src.models.yolo_detector import YOLODetector
        from src.models.severity_estimator import SeverityEstimator
        
        self.device = device
        self.batch_size = batch_size
        self.detector = YOLODetector(model_path, device, use_half=use_half)
        self.severity_estimator = SeverityEstimator()
        
        self.inference_times = []
    
    def infer_single(self, image: np.ndarray) -> Dict:
        """
        Run inference on single image.
        
        Args:
            image: Input image
            
        Returns:
            Detections with severity
        """
        start_time = time.time()
        
        # Detection
        result = self.detector.detect(image)
        detections = result['detections']
        
        # Severity estimation
        detections = self.severity_estimator.estimate_severity(image, detections)
        
        # Ranking
        detections = self.severity_estimator.get_priority_ranking(detections)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'detections': detections,
            'inference_time_ms': inference_time * 1000
        }
    
    def infer_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Batch inference."""
        results = []
        
        for image in images:
            result = self.infer_single(image)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'min_inference_time_ms': 0,
                'fps': 0
            }
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'fps': float(1000.0 / np.mean(times_ms))
        }
    
    def reset_statistics(self):
        """Reset inference statistics."""
        self.inference_times = []
