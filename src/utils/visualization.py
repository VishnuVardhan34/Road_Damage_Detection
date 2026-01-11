"""
Utility functions for visualization and geometry.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class BboxVisualizer:
    """Visualize bounding boxes and detections on images."""
    
    COLORS = {
        'Crack': (0, 255, 0),           # Green
        'Pothole': (0, 0, 255),         # Red
        'Rutting': (255, 0, 0),         # Blue
        'Patch': (0, 165, 255),         # Orange
        'Lane_Wear': (255, 255, 0),     # Cyan
        'Manhole': (255, 0, 255)        # Magenta
    }
    
    SEVERITY_COLORS = {
        'Good': (0, 255, 0),           # Green
        'Moderate': (0, 255, 255),     # Yellow
        'Severe': (0, 165, 255),       # Orange
        'Critical': (0, 0, 255)        # Red
    }
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict],
                       color_by_severity: bool = False,
                       show_confidence: bool = True,
                       show_severity: bool = True) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image (BGR)
            detections: List of detections
            color_by_severity: Use severity color instead of class color
            show_confidence: Display confidence scores
            show_severity: Display severity scores
            
        Returns:
            Annotated image
        """
        image = image.copy()
        h, w = image.shape[:2]
        
        for det in detections:
            # Get bbox coordinates
            bbox = det['bbox']  # normalized
            x_center = int(bbox[0] * w)
            y_center = int(bbox[1] * h)
            width = int(bbox[2] * w)
            height = int(bbox[3] * h)
            
            x1 = x_center - width // 2
            y1 = y_center - height // 2
            x2 = x_center + width // 2
            y2 = y_center + height // 2
            
            # Ensure coordinates are within image
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Select color
            if color_by_severity and 'severity_category' in det:
                color = BboxVisualizer.SEVERITY_COLORS.get(
                    det['severity_category'], (255, 255, 255)
                )
            else:
                color = BboxVisualizer.COLORS.get(
                    det['class_name'], (255, 255, 255)
                )
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [det['class_name']]
            if show_confidence:
                label_parts.append(f"C:{det['confidence']:.2f}")
            if show_severity and 'severity' in det:
                label_parts.append(f"S:{det['severity']:.1f}")
            
            label = ' '.join(label_parts)
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            label_y = max(20, y1 - 5)
            label_x = max(5, x1)
            
            cv2.rectangle(image,
                         (label_x, label_y - text_size[1] - 5),
                         (label_x + text_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(image, label, (label_x + 2, label_y - 2),
                       font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    @staticmethod
    def draw_severity_heatmap(image: np.ndarray, detections: List[Dict],
                             alpha: float = 0.3) -> np.ndarray:
        """
        Create severity heatmap overlay.
        
        Args:
            image: Input image
            detections: List of detections with severity
            alpha: Transparency
            
        Returns:
            Image with heatmap overlay
        """
        heatmap = np.zeros_like(image, dtype=np.float32)
        h, w = image.shape[:2]
        
        for det in detections:
            severity = det.get('severity', 0)
            bbox = det['bbox']
            
            x_center = int(bbox[0] * w)
            y_center = int(bbox[1] * h)
            width = int(bbox[2] * w)
            height = int(bbox[3] * h)
            
            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(w, x_center + width // 2)
            y2 = min(h, y_center + height // 2)
            
            # Color based on severity
            if severity < 25:
                color = (0, 255, 0)  # Green
            elif severity < 50:
                color = (0, 255, 255)  # Yellow
            elif severity < 75:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            intensity = severity / 100.0
            for c in range(3):
                heatmap[y1:y2, x1:x2, c] = color[c] * intensity
        
        # Blend with original
        result = cv2.addWeighted(image, 1 - alpha, heatmap.astype(np.uint8), alpha, 0)
        return result


class GeometryUtils:
    """Geometry utility functions."""
    
    @staticmethod
    def calculate_bbox_area(bbox: Tuple[float, float, float, float],
                           image_shape: Tuple[int, int]) -> int:
        """
        Calculate bbox area in pixels.
        
        Args:
            bbox: Normalized bbox (x_center, y_center, width, height)
            image_shape: Image shape (height, width)
            
        Returns:
            Area in pixels
        """
        h, w = image_shape[:2]
        width_px = bbox[2] * w
        height_px = bbox[3] * h
        return int(width_px * height_px)
    
    @staticmethod
    def bbox_to_corners(bbox: Tuple[float, float, float, float],
                       image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized bbox to pixel coordinates.
        
        Returns:
            (x1, y1, x2, y2) in pixels
        """
        x_center, y_center, width, height = bbox
        
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)
        
        return max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
