"""
Severity estimation module for road damage.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SeverityEstimator:
    """
    Estimate damage severity based on bounding box area, damage density, and texture roughness.
    
    Severity Formula:
    Severity = 0.45 * AreaRatio + 0.30 * DamageCountNormalization + 0.25 * TextureRoughness
    
    Severity Categories:
    - 0-25: Good
    - 25-50: Moderate
    - 50-75: Severe
    - 75-100: Critical
    """
    
    SEVERITY_CATEGORIES = {
        'Good': (0, 25),
        'Moderate': (25, 50),
        'Severe': (50, 75),
        'Critical': (75, 100)
    }
    
    WEIGHTS = {
        'area_ratio': 0.45,
        'damage_count': 0.30,
        'texture_roughness': 0.25
    }
    
    def __init__(self, weights: Dict[str, float] = None,
                 sobel_kernel_size: int = 3,
                 roughness_percentile: int = 95,
                 min_damage_area: int = 50):
        """
        Initialize severity estimator.
        
        Args:
            weights: Custom weights for severity components
            sobel_kernel_size: Kernel size for Sobel edge detection
            roughness_percentile: Percentile for texture roughness
            min_damage_area: Minimum pixel area to consider as damage
        """
        if weights:
            self.weights = weights
        else:
            self.weights = self.WEIGHTS.copy()
        
        self.sobel_kernel_size = sobel_kernel_size
        self.roughness_percentile = roughness_percentile
        self.min_damage_area = min_damage_area
    
    def estimate_severity(self, image: np.ndarray, 
                         detections: List[Dict]) -> List[Dict]:
        """
        Estimate severity for detected damages.
        
        Args:
            image: Input image (H, W, C)
            detections: List of detections with bbox in format
                       {class_id, bbox: (x_center, y_center, w, h), confidence}
            
        Returns:
            List of detections with added severity scores and categories
        """
        if len(detections) == 0:
            return detections
        
        image_h, image_w = image.shape[:2]
        
        # Compute area ratio
        area_ratios = []
        for det in detections:
            bbox = det['bbox']
            # Convert normalized coords to pixels
            w_px = bbox[2] * image_w
            h_px = bbox[3] * image_h
            area = w_px * h_px
            area_ratio = area / (image_h * image_w)
            area_ratios.append(area_ratio)
        
        # Normalize area ratios
        max_area_ratio = max(area_ratios) if area_ratios else 1.0
        area_ratios_norm = [ar / max_area_ratio for ar in area_ratios]
        
        # Compute damage count normalization
        damage_count = len(detections)
        # Normalize by typical max detections
        damage_count_norm = min(damage_count / 20.0, 1.0)  # Assume max ~20 damages
        
        # Compute texture roughness for entire image region with damages
        roughness_scores = []
        for det in detections:
            bbox = det['bbox']
            # Extract region of interest
            x_center = int(bbox[0] * image_w)
            y_center = int(bbox[1] * image_h)
            w = int(bbox[2] * image_w)
            h = int(bbox[3] * image_h)
            
            x1 = max(0, x_center - w // 2)
            y1 = max(0, y_center - h // 2)
            x2 = min(image_w, x_center + w // 2)
            y2 = min(image_h, y_center + h // 2)
            
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                roughness = self._compute_roughness(roi)
                roughness_scores.append(roughness)
            else:
                roughness_scores.append(0.0)
        
        # Compute severity for each detection
        detections_with_severity = []
        for i, det in enumerate(detections):
            # Weighted combination
            severity = (
                self.weights['area_ratio'] * area_ratios_norm[i] * 100 +
                self.weights['damage_count'] * damage_count_norm * 100 +
                self.weights['texture_roughness'] * roughness_scores[i] * 100
            )
            
            severity = np.clip(severity, 0, 100)
            category = self._get_severity_category(severity)
            
            det['severity'] = float(severity)
            det['severity_category'] = category
            detections_with_severity.append(det)
        
        return detections_with_severity
    
    def _compute_roughness(self, image: np.ndarray) -> float:
        """
        Compute texture roughness using Sobel edge detection.
        
        Args:
            image: Image region (H, W, C)
            
        Returns:
            Roughness score [0-1]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image.size == 0:
            return 0.0
        
        # Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, 
                          ksize=self.sobel_kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, 
                          ksize=self.sobel_kernel_size)
        
        edge_map = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        edge_map = edge_map / (255.0 * np.sqrt(2))  # Max possible gradient
        
        # Compute percentile-based roughness
        roughness = np.percentile(edge_map, self.roughness_percentile) / 255.0
        
        return float(np.clip(roughness, 0, 1))
    
    def _get_severity_category(self, severity: float) -> str:
        """Get severity category name from score."""
        for category, (low, high) in self.SEVERITY_CATEGORIES.items():
            if low <= severity < high:
                return category
        return 'Critical'
    
    def get_priority_ranking(self, detections: List[Dict]) -> List[Dict]:
        """
        Rank detections by severity for maintenance prioritization.
        
        Args:
            detections: List of detections with severity scores
            
        Returns:
            Sorted list of detections (highest severity first)
        """
        return sorted(detections, key=lambda x: x.get('severity', 0), reverse=True)
    
    def generate_severity_report(self, detections: List[Dict]) -> Dict:
        """
        Generate severity statistics report.
        
        Args:
            detections: List of detections with severity scores
            
        Returns:
            Dictionary with severity statistics
        """
        if not detections:
            return {
                'total_damages': 0,
                'average_severity': 0,
                'critical_count': 0,
                'severe_count': 0,
                'moderate_count': 0,
                'good_count': 0
            }
        
        severities = [det['severity'] for det in detections]
        categories = [det['severity_category'] for det in detections]
        
        return {
            'total_damages': len(detections),
            'average_severity': float(np.mean(severities)),
            'max_severity': float(np.max(severities)),
            'min_severity': float(np.min(severities)),
            'std_severity': float(np.std(severities)),
            'critical_count': sum(1 for c in categories if c == 'Critical'),
            'severe_count': sum(1 for c in categories if c == 'Severe'),
            'moderate_count': sum(1 for c in categories if c == 'Moderate'),
            'good_count': sum(1 for c in categories if c == 'Good')
        }
