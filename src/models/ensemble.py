"""
Model ensemble for improved detection accuracy.
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DetectionEnsemble:
    """Ensemble multiple detectors for improved accuracy."""
    
    def __init__(self, detectors: List, weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            detectors: List of detector objects
            weights: Weights for each detector (default: equal)
        """
        self.detectors = detectors
        
        if weights is None:
            weights = [1.0 / len(detectors)] * len(detectors)
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        logger.info(f"Initialized ensemble with {len(detectors)} detectors")
    
    def detect(self, image, image_size: int = 1024) -> Dict:
        """
        Run ensemble detection.
        
        Args:
            image: Input image
            image_size: Input size
            
        Returns:
            Merged detections
        """
        all_detections = []
        
        for detector, weight in zip(self.detectors, self.weights):
            result = detector.detect(image, image_size)
            
            # Add weight to each detection
            for det in result['detections']:
                det['weight'] = weight
                all_detections.append(det)
        
        # Merge detections by NMS
        merged = self._merge_detections(all_detections)
        
        return {
            'image': image,
            'detections': merged,
            'num_detections': len(merged)
        }
    
    def _merge_detections(self, detections: List[Dict],
                         iou_threshold: float = 0.5) -> List[Dict]:
        """
        Merge overlapping detections from multiple models.
        
        Args:
            detections: List of detections with weights
            iou_threshold: IoU threshold for merging
            
        Returns:
            Merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, 
                          key=lambda x: x['confidence'] * x['weight'],
                          reverse=True)
        
        merged = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            # Collect similar detections
            group = [det]
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                # Check if same class and high IoU
                if det['class_id'] == detections[j]['class_id']:
                    iou = self._compute_iou(det['bbox'], detections[j]['bbox'])
                    if iou > iou_threshold:
                        group.append(detections[j])
                        used.add(j)
            
            # Merge group
            merged_det = self._merge_detection_group(group)
            merged.append(merged_det)
        
        return merged
    
    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes (normalized coordinates)."""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _merge_detection_group(group: List[Dict]) -> Dict:
        """Merge a group of similar detections."""
        # Weighted average confidence
        total_weight = sum(det['weight'] for det in group)
        avg_confidence = sum(det['confidence'] * det['weight'] 
                           for det in group) / total_weight
        
        # Weighted average bbox
        avg_bbox = [
            sum(det['bbox'][i] * det['weight'] for det in group) / total_weight
            for i in range(4)
        ]
        
        return {
            'class_id': group[0]['class_id'],
            'class_name': group[0]['class_name'],
            'bbox': tuple(avg_bbox),
            'confidence': avg_confidence,
            'num_models': len(group)
        }
