"""
Script to convert Pascal VOC annotations to YOLO format.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data.converters import VOCToYOLOConverter, create_dataset_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Pascal VOC annotations to YOLO format'
    )
    parser.add_argument('--voc_dir', type=str, required=True,
                       help='Directory containing VOC XML annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for YOLO txt files')
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON file with class mapping (optional)')
    
    args = parser.parse_args()
    
    # Default class mapping (RDD2022)
    class_mapping = {
        'D00': 0,  # Longitudinal crack
        'D01': 0,  # Transverse crack
        'D10': 0,  # Alligator crack
        'D11': 1,  # Pothole
        'D20': 2,  # Rutting
        'D40': 3,  # Patch
        'D43': 4,  # Lane marking wear
        'D44': 5,  # Manhole
    }
    
    # Create converter
    converter = VOCToYOLOConverter(class_mapping)
    
    # Convert annotations
    logger.info(f"Converting VOC annotations from {args.voc_dir}")
    stats = converter.batch_convert(args.voc_dir, args.output_dir)
    
    logger.info(f"Conversion completed:")
    logger.info(f"  - Successful: {stats['success']}")
    logger.info(f"  - Failed: {stats['failed']}")
    logger.info(f"  - Output directory: {args.output_dir}")
    
    # Create dataset.yaml
    output_yaml = Path(args.output_dir).parent / 'dataset.yaml'
    class_names = ['Crack', 'Pothole', 'Rutting', 'Patch', 'Lane_Wear', 'Manhole']
    
    create_dataset_yaml(
        str(output_yaml),
        str(Path(args.output_dir).parent),
        num_classes=len(class_names),
        class_names=class_names
    )
    
    logger.info(f"Dataset YAML created: {output_yaml}")


if __name__ == '__main__':
    main()
