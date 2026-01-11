"""
Custom dataset classes for road damage detection.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class RDD2022Dataset(Dataset):
    """
    Road Damage Detection 2022 Dataset.
    
    Expects:
    - images/ directory with image files
    - labels/ directory with YOLO format txt files (one per image)
    - dataset.yaml with class information
    """
    
    def __init__(self, image_dir: str, labels_dir: str, 
                 image_size: int = 1024, 
                 augment: bool = True,
                 augmentation_config: Optional[Dict] = None):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            image_size: Input image size
            augment: Whether to apply augmentations
            augmentation_config: Dictionary with augmentation parameters
        """
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augment = augment
        
        # Get all image files
        self.image_files = self._get_image_files()
        logger.info(f"Loaded {len(self.image_files)} images from {image_dir}")
        
        # Setup augmentations
        self.transform = self._get_transforms(augment, augmentation_config)
    
    def _get_image_files(self) -> List[str]:
        """Get all image files in directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(self.image_dir).glob(f'*{ext}'))
            image_files.extend(Path(self.image_dir).glob(f'*{ext.upper()}'))
        
        return sorted([f.stem for f in image_files])
    
    def _get_transforms(self, augment: bool, 
                       config: Optional[Dict]) -> A.Compose:
        """Get albumentations transforms."""
        
        if config is None:
            config = {}
        
        transforms = []
        
        if augment:
            # Mosaic-like augmentation using albumentation
            transforms.extend([
                A.HorizontalFlip(p=config.get('fliplr', 0.5)),
                A.VerticalFlip(p=config.get('flipud', 0.5)),
                A.Rotate(limit=config.get('degrees', 10), p=0.5),
                A.Perspective(scale=config.get('perspective', 0.0), p=0.5),
                A.Affine(shear=config.get('shear', 0), p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=config.get('translate', 0.1),
                    scale_limit=config.get('scale', 0.5),
                    rotate_limit=config.get('degrees', 10),
                    p=0.5
                ),
                A.GaussNoise(p=config.get('gaussian_blur', 0.2)),
                A.MotionBlur(blur_limit=5, p=config.get('motion_blur', 0.3)),
                A.RandomShadow(p=config.get('shadows', 0.3)),
                A.ColorJitter(
                    brightness=config.get('hsv_v', 0.4),
                    contrast=config.get('hsv_s', 0.7),
                    saturation=config.get('hsv_s', 0.7),
                    hue=config.get('hsv_h', 0.015),
                    p=0.5
                ),
            ])
        
        # Resize and normalize
        transforms.extend([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo', min_visibility=0.3))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item by index.
        
        Returns:
            Dictionary with 'image' and 'targets' keys
        """
        image_name = self.image_files[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_name + '.jpg')
        if not os.path.exists(image_path):
            # Try other extensions
            for ext in ['.png', '.jpeg', '.bmp']:
                alt_path = os.path.join(self.image_dir, image_name + ext)
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return self._get_dummy_item()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.labels_dir, image_name + '.txt')
        bboxes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append([x_center, y_center, width, height, class_id])
        
        bboxes = np.array(bboxes, dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes[:, :4] if len(bboxes) > 0 else [],
                class_labels=bboxes[:, 4].tolist() if len(bboxes) > 0 else []
            )
            image = transformed['image']
            
            # Reconstruct bboxes with class labels
            if len(transformed['bboxes']) > 0:
                bboxes = np.column_stack([
                    transformed['bboxes'],
                    transformed['class_labels']
                ])
            else:
                bboxes = np.empty((0, 5), dtype=np.float32)
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'image_name': image_name
        }
    
    def _get_dummy_item(self) -> Dict:
        """Return dummy item when loading fails."""
        return {
            'image': torch.zeros((3, self.image_size, self.image_size)),
            'bboxes': torch.zeros((0, 5)),
            'image_name': 'dummy'
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for DataLoader."""
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'image_names': image_names
    }
