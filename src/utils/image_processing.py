"""
Image processing and augmentation utilities.
"""

import cv2
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities for road damage detection."""
    
    @staticmethod
    def resize_image(image: np.ndarray, size: int = 1024,
                     maintain_aspect: bool = True) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize image to target size.
        
        Args:
            image: Input image (H, W, C)
            size: Target size
            maintain_aspect: Whether to maintain aspect ratio with padding
            
        Returns:
            Resized image and scale factors (scale_x, scale_y)
        """
        h, w = image.shape[:2]
        
        if maintain_aspect:
            scale = size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to size
            canvas = np.zeros((size, size, 3), dtype=image.dtype)
            y_offset = (size - new_h) // 2
            x_offset = (size - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas, (scale, scale)
        else:
            scale_x = size / w
            scale_y = size / h
            resized = cv2.resize(image, (size, size))
            return resized, (scale_x, scale_y)
    
    @staticmethod
    def normalize_image(image: np.ndarray,
                       mean: List[float] = None,
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image with mean and std.
        
        Args:
            image: Input image (H, W, C) in range [0, 255]
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray,
                         mean: List[float] = None,
                         std: List[float] = None) -> np.ndarray:
        """Reverse normalization."""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        for i in range(3):
            image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        
        return np.clip((image * 255).astype(np.uint8), 0, 255)
    
    @staticmethod
    def compute_image_statistics(image: np.ndarray) -> dict:
        """Compute statistics of image."""
        return {
            'mean': image.mean(axis=(0, 1)),
            'std': image.std(axis=(0, 1)),
            'min': image.min(axis=(0, 1)),
            'max': image.max(axis=(0, 1))
        }


class TextureAnalyzer:
    """Analyze texture roughness and surface characteristics."""
    
    @staticmethod
    def sobel_edge_detection(image: np.ndarray, 
                            kernel_size: int = 3) -> np.ndarray:
        """
        Compute edge map using Sobel operator.
        
        Args:
            image: Input grayscale image
            kernel_size: Sobel kernel size
            
        Returns:
            Edge map
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        edge_map = np.sqrt(sobelx**2 + sobely**2)
        edge_map = np.uint8(np.clip(edge_map, 0, 255))
        
        return edge_map
    
    @staticmethod
    def compute_texture_roughness(image: np.ndarray,
                                  percentile: int = 95) -> float:
        """
        Compute texture roughness using edge detection.
        
        Args:
            image: Input image
            percentile: Percentile for thresholding edges
            
        Returns:
            Roughness score [0-1]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = TextureAnalyzer.sobel_edge_detection(image)
        
        # Normalize and compute roughness
        roughness = np.percentile(edges, percentile) / 255.0
        
        return float(roughness)
    
    @staticmethod
    def compute_local_variance(image: np.ndarray,
                               window_size: int = 7) -> np.ndarray:
        """
        Compute local variance (texture roughness map).
        
        Args:
            image: Input grayscale image
            window_size: Size of local window
            
        Returns:
            Variance map
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float32)
        
        # Compute mean
        mean = cv2.blur(image, (window_size, window_size))
        
        # Compute squared mean
        sq_image = image ** 2
        sq_mean = cv2.blur(sq_image, (window_size, window_size))
        
        # Variance
        variance = sq_mean - (mean ** 2)
        variance = np.sqrt(np.maximum(variance, 0))
        
        return variance.astype(np.uint8)
