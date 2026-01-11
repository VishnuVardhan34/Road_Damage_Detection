"""
Data utility functions for Pascal VOC to YOLO format conversion and dataset handling.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


class VOCToYOLOConverter:
    """Convert Pascal VOC annotations to YOLO format."""
    
    def __init__(self, class_mapping: Dict[str, int]):
        """
        Initialize converter with class mapping.
        
        Args:
            class_mapping: Dictionary mapping class names to indices
        """
        self.class_mapping = class_mapping
    
    def parse_voc_xml(self, xml_file: str) -> Dict:
        """Parse Pascal VOC XML annotation file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        data = {
            'filename': root.find('filename').text,
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text),
            'objects': []
        }
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            data['objects'].append({
                'class': name,
                'bbox': (xmin, ymin, xmax, ymax)
            })
        
        return data
    
    def voc_to_yolo_bbox(self, bbox: Tuple, image_width: int, 
                        image_height: int) -> Tuple[float, float, float, float]:
        """
        Convert VOC bbox (xmin, ymin, xmax, ymax) to YOLO format 
        (center_x, center_y, width, height) normalized [0-1].
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Convert to center coordinates
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Normalize
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        
        return center_x, center_y, width, height
    
    def convert_annotation(self, voc_xml_file: str, output_txt_file: str) -> bool:
        """
        Convert single annotation file from VOC to YOLO format.
        
        Args:
            voc_xml_file: Path to VOC XML file
            output_txt_file: Path to output YOLO txt file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.parse_voc_xml(voc_xml_file)
            
            with open(output_txt_file, 'w') as f:
                for obj in data['objects']:
                    class_name = obj['class']
                    
                    if class_name not in self.class_mapping:
                        logger.warning(f"Unknown class: {class_name}")
                        continue
                    
                    class_id = self.class_mapping[class_name]
                    bbox = obj['bbox']
                    
                    yolo_bbox = self.voc_to_yolo_bbox(
                        bbox, data['width'], data['height']
                    )
                    
                    # YOLO format: class_id center_x center_y width height
                    line = f"{class_id} " + " ".join(f"{v:.6f}" for v in yolo_bbox)
                    f.write(line + "\n")
            
            return True
        except Exception as e:
            logger.error(f"Error converting {voc_xml_file}: {e}")
            return False
    
    def batch_convert(self, input_dir: str, output_dir: str) -> Dict[str, int]:
        """
        Batch convert VOC annotations to YOLO format.
        
        Args:
            input_dir: Directory containing VOC XML files
            output_dir: Output directory for YOLO txt files
            
        Returns:
            Dictionary with conversion statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        xml_files = list(Path(input_dir).glob('*.xml'))
        logger.info(f"Found {len(xml_files)} XML files to convert")
        
        for xml_file in xml_files:
            txt_file = os.path.join(output_dir, xml_file.stem + '.txt')
            if self.convert_annotation(str(xml_file), txt_file):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Conversion complete: {stats['success']} succeeded, "
                   f"{stats['failed']} failed")
        
        return stats


class DataSplitter:
    """Split dataset into train/val/test sets."""
    
    @staticmethod
    def create_splits(image_dir: str, output_dir: str, 
                     train_split: float = 0.7,
                     val_split: float = 0.2,
                     test_split: float = 0.1,
                     seed: int = 42) -> Dict[str, List[str]]:
        """
        Create train/val/test splits and save to files.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save split files
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with file lists for each split
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        image_files = [f.stem for f in image_files]
        
        # Shuffle with seed
        np.random.seed(seed)
        np.random.shuffle(image_files)
        
        # Split
        total = len(image_files)
        train_idx = int(total * train_split)
        val_idx = train_idx + int(total * val_split)
        
        splits = {
            'train': image_files[:train_idx],
            'val': image_files[train_idx:val_idx],
            'test': image_files[val_idx:]
        }
        
        # Save to files
        for split_name, files in splits.items():
            output_file = os.path.join(output_dir, f'{split_name}.txt')
            with open(output_file, 'w') as f:
                for file in files:
                    f.write(file + '\n')
            
            logger.info(f"{split_name}: {len(files)} images")
        
        return splits


def create_dataset_yaml(output_path: str, data_dir: str, 
                       num_classes: int, class_names: List[str]):
    """
    Create dataset.yaml file for YOLO training.
    
    Args:
        output_path: Path to save dataset.yaml
        data_dir: Directory containing dataset
        num_classes: Number of classes
        class_names: List of class names
    """
    dataset_yaml = {
        'path': data_dir,
        'train': 'splits/train.txt',
        'val': 'splits/val.txt',
        'test': 'splits/test.txt',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        for key, value in dataset_yaml.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, int):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Dataset YAML created: {output_path}")
