"""
Unit tests for core modules.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path


class TestDataConverters:
    """Test Pascal VOC to YOLO conversion."""
    
    def test_yolo_bbox_normalization(self):
        """Test bbox normalization."""
        from src.data.converters import VOCToYOLOConverter
        
        converter = VOCToYOLOConverter({'test': 0})
        
        # Test bbox conversion
        bbox = (100, 100, 200, 200)  # xmin, ymin, xmax, ymax
        image_w, image_h = 1024, 1024
        
        result = converter.voc_to_yolo_bbox(bbox, image_w, image_h)
        
        # Expected: center at (150/1024, 150/1024), width/height 100/1024
        assert len(result) == 4
        assert 0 <= result[0] <= 1  # x_center
        assert 0 <= result[1] <= 1  # y_center
        assert 0 <= result[2] <= 1  # width
        assert 0 <= result[3] <= 1  # height


class TestSeverityEstimator:
    """Test severity estimation."""
    
    def test_severity_calculation(self):
        """Test severity score calculation."""
        from src.models.severity_estimator import SeverityEstimator
        
        estimator = SeverityEstimator()
        
        # Create dummy image
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Create dummy detections
        detections = [
            {
                'class_id': 0,
                'class_name': 'Crack',
                'bbox': (0.5, 0.5, 0.2, 0.2),
                'confidence': 0.9
            }
        ]
        
        # Estimate severity
        result = estimator.estimate_severity(image, detections)
        
        assert len(result) == 1
        assert 'severity' in result[0]
        assert 0 <= result[0]['severity'] <= 100
        assert 'severity_category' in result[0]
    
    def test_severity_categories(self):
        """Test severity category assignment."""
        from src.models.severity_estimator import SeverityEstimator
        
        estimator = SeverityEstimator()
        
        assert estimator._get_severity_category(10) == 'Good'
        assert estimator._get_severity_category(35) == 'Moderate'
        assert estimator._get_severity_category(60) == 'Severe'
        assert estimator._get_severity_category(85) == 'Critical'


class TestImageProcessing:
    """Test image processing utilities."""
    
    def test_image_resize(self):
        """Test image resizing."""
        from src.utils.image_processing import ImageProcessor
        
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        resized, scales = ImageProcessor.resize_image(image, size=1024, maintain_aspect=True)
        
        assert resized.shape == (1024, 1024, 3)
        assert all(0 < s <= 2 for s in scales)
    
    def test_texture_roughness(self):
        """Test texture roughness computation."""
        from src.utils.image_processing import TextureAnalyzer
        
        # Create smooth image (low roughness)
        smooth_image = np.ones((100, 100), dtype=np.uint8) * 128
        roughness_smooth = TextureAnalyzer.compute_texture_roughness(smooth_image)
        
        # Create rough image (high roughness)
        rough_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        roughness_rough = TextureAnalyzer.compute_texture_roughness(rough_image)
        
        # Rough image should have higher roughness
        assert roughness_rough > roughness_smooth


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_iou_calculation(self):
        """Test IoU calculation."""
        from src.utils.metrics import MetricsCalculator
        
        box1 = np.array([0.5, 0.5, 0.4, 0.4])  # x_center, y_center, width, height
        box2 = np.array([0.5, 0.5, 0.4, 0.4])  # Same box
        
        iou = MetricsCalculator.compute_iou(box1, box2)
        
        assert abs(iou - 1.0) < 1e-6  # Should be 1.0
    
    def test_iou_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        from src.utils.metrics import MetricsCalculator
        
        box1 = np.array([0.2, 0.2, 0.2, 0.2])
        box2 = np.array([0.8, 0.8, 0.2, 0.2])  # Far apart
        
        iou = MetricsCalculator.compute_iou(box1, box2)
        
        assert iou == 0.0


class TestVisualization:
    """Test visualization utilities."""
    
    def test_bbox_drawing(self):
        """Test bbox visualization."""
        from src.utils.visualization import BboxVisualizer
        
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        detections = [
            {
                'class_name': 'Pothole',
                'confidence': 0.95,
                'severity': 75.5,
                'severity_category': 'Severe',
                'bbox': (0.5, 0.5, 0.2, 0.2)
            }
        ]
        
        result = BboxVisualizer.draw_detections(
            image, detections, 
            color_by_severity=True,
            show_confidence=True,
            show_severity=True
        )
        
        assert result.shape == image.shape
        # Check that image has been modified (contains drawn elements)
        assert not np.array_equal(result, image)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
