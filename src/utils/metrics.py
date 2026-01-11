"""
Evaluation metrics for object detection.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate detection metrics."""
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        Boxes are in format (x_center, y_center, width, height) normalized [0-1].
        """
        # Convert to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        return float(iou)
    
    @staticmethod
    def compute_ap(predictions: List[Dict], targets: List[Dict],
                   iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision for a single class.
        
        Args:
            predictions: List of predictions {bbox, confidence}
            targets: List of ground truth {bbox}
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Average Precision score
        """
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        targets_matched = set()
        
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_target_idx = -1
            
            for j, target in enumerate(targets):
                if j in targets_matched:
                    continue
                
                iou = MetricsCalculator.compute_iou(
                    np.array(pred['bbox']),
                    np.array(target['bbox'])
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp[i] = 1
                targets_matched.add(best_target_idx)
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(targets) if len(targets) > 0 else np.zeros_like(tp_cumsum)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        return float(ap)
    
    @staticmethod
    def compute_map(predictions_per_class: Dict[int, List[Dict]],
                   targets_per_class: Dict[int, List[Dict]],
                   iou_threshold: float = 0.5) -> float:
        """Compute mean Average Precision across all classes."""
        aps = []
        
        for class_id in targets_per_class.keys():
            if class_id not in predictions_per_class:
                predictions = []
            else:
                predictions = predictions_per_class[class_id]
            
            targets = targets_per_class[class_id]
            
            if len(targets) == 0:
                continue
            
            ap = MetricsCalculator.compute_ap(predictions, targets, iou_threshold)
            aps.append(ap)
        
        return float(np.mean(aps)) if len(aps) > 0 else 0.0
    
    @staticmethod
    def compute_confusion_matrix(predictions: List[int],
                                targets: List[int],
                                num_classes: int) -> np.ndarray:
        """Compute confusion matrix."""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for pred, target in zip(predictions, targets):
            if 0 <= pred < num_classes and 0 <= target < num_classes:
                cm[target, pred] += 1
        
        return cm
    
    @staticmethod
    def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(precision), float(recall), float(f1)
