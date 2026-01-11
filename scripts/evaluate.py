"""
Evaluation script for model assessment.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, data_dir: str, 
                  image_size: int = 1024) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to YOLO model
        data_dir: Path to dataset directory
        image_size: Input image size
        
    Returns:
        Evaluation metrics
    """
    from src.models.yolo_detector import YOLODetector
    from src.utils.metrics import MetricsCalculator
    
    logger.info("Loading model...")
    detector = YOLODetector(model_path, device='cuda')
    
    # Load test set
    test_images_dir = os.path.join(data_dir, 'images', 'test')
    test_labels_dir = os.path.join(data_dir, 'labels', 'test')
    
    if not os.path.exists(test_images_dir):
        logger.error(f"Test images not found: {test_images_dir}")
        return {}
    
    # Get all test images
    test_images = list(Path(test_images_dir).glob('*.jpg'))
    test_images.extend(Path(test_images_dir).glob('*.png'))
    
    logger.info(f"Evaluating on {len(test_images)} test images...")
    
    all_predictions = {}
    all_targets = {}
    
    for img_path in tqdm(test_images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Get predictions
        result = detector.detect(image, image_size)
        predictions = result['detections']
        
        # Get ground truth
        label_path = os.path.join(test_labels_dir, img_path.stem + '.txt')
        targets = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = tuple(float(p) for p in parts[1:5])
                        targets.append({
                            'class_id': class_id,
                            'bbox': bbox
                        })
        
        # Organize by class
        for det in predictions:
            cls_id = det['class_id']
            if cls_id not in all_predictions:
                all_predictions[cls_id] = []
            all_predictions[cls_id].append({
                'bbox': det['bbox'],
                'confidence': det['confidence']
            })
        
        for target in targets:
            cls_id = target['class_id']
            if cls_id not in all_targets:
                all_targets[cls_id] = []
            all_targets[cls_id].append(target)
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    map_score = MetricsCalculator.compute_map(
        all_predictions, all_targets, iou_threshold=0.5
    )
    
    # Per-class metrics
    class_names = [
        'Crack', 'Pothole', 'Rutting', 'Patch', 'Lane_Wear', 'Manhole'
    ]
    
    per_class_ap = {}
    for class_id in all_targets.keys():
        if class_id not in all_predictions:
            predictions = []
        else:
            predictions = all_predictions[class_id]
        
        targets = all_targets[class_id]
        ap = MetricsCalculator.compute_ap(predictions, targets)
        per_class_ap[class_names[class_id]] = float(ap)
    
    results = {
        'mAP@0.5': float(map_score),
        'per_class_ap': per_class_ap,
        'num_test_images': len(test_images),
        'num_total_detections': sum(len(v) for v in all_predictions.values()),
        'num_total_targets': sum(len(v) for v in all_targets.values())
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate road damage detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model')
    parser.add_argument('--data', type=str, default='data/processed',
                       help='Path to dataset')
    parser.add_argument('--output', type=str, default='results/evaluation.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_model(args.model, args.data)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    logger.info(f"mAP@0.5: {results.get('mAP@0.5', 0):.4f}")
    logger.info("\nPer-class AP:")
    for class_name, ap in results.get('per_class_ap', {}).items():
        logger.info(f"  {class_name}: {ap:.4f}")
    
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
