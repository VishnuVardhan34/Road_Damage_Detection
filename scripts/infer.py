"""
Inference script for road damage detection.
"""

import argparse
import logging
import os
import sys
import cv2
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def infer_image(model_path: str, image_path: str, output_dir: str,
               config_path: str = 'config/inference_config.yaml'):
    """
    Run inference on single image.
    
    Args:
        model_path: Path to YOLO model
        image_path: Path to input image
        output_dir: Directory to save annotated image
        config_path: Path to inference config
    """
    from src.models.yolo_detector import YOLODetector
    from src.models.severity_estimator import SeverityEstimator
    from src.utils.visualization import BboxVisualizer
    
    logger.info("Loading model...")
    detector = YOLODetector(model_path)
    severity_estimator = SeverityEstimator()
    
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    logger.info("Running detection...")
    result = detector.detect(image)
    detections = result['detections']
    
    logger.info(f"Found {len(detections)} potential damages")
    
    # Estimate severity
    detections = severity_estimator.estimate_severity(image, detections)
    
    # Rank by severity
    detections = severity_estimator.get_priority_ranking(detections)
    
    # Generate report
    report = severity_estimator.generate_severity_report(detections)
    
    logger.info("Severity Report:")
    logger.info(f"  - Total damages: {report['total_damages']}")
    logger.info(f"  - Average severity: {report['average_severity']:.2f}")
    logger.info(f"  - Critical: {report['critical_count']}")
    logger.info(f"  - Severe: {report['severe_count']}")
    logger.info(f"  - Moderate: {report['moderate_count']}")
    logger.info(f"  - Good: {report['good_count']}")
    
    # Visualize
    annotated = BboxVisualizer.draw_detections(
        image, detections,
        color_by_severity=True,
        show_confidence=True,
        show_severity=True
    )
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'annotated.jpg')
    cv2.imwrite(output_path, annotated)
    
    logger.info(f"Annotated image saved: {output_path}")
    
    # Print detections
    logger.info("\nDetections (ranked by severity):")
    for i, det in enumerate(detections, 1):
        logger.info(
            f"{i}. {det['class_name']} - "
            f"Severity: {det['severity']:.1f} ({det['severity_category']}) - "
            f"Confidence: {det['confidence']:.3f}"
        )


def infer_directory(model_path: str, image_dir: str, output_dir: str):
    """
    Run inference on all images in directory.
    
    Args:
        model_path: Path to YOLO model
        image_dir: Directory containing images
        output_dir: Directory to save results
    """
    from src.models.yolo_detector import YOLODetector
    from src.models.severity_estimator import SeverityEstimator
    from src.utils.visualization import BboxVisualizer
    
    logger.info("Loading model...")
    detector = YOLODetector(model_path)
    severity_estimator = SeverityEstimator()
    
    # Get all images
    image_files = list(Path(image_dir).glob('*.jpg'))
    image_files.extend(Path(image_dir).glob('*.png'))
    
    logger.info(f"Found {len(image_files)} images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"[{i}/{len(image_files)}] Processing {image_path.name}...")
        
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load: {image_path}")
            continue
        
        # Detect
        result = detector.detect(image)
        detections = result['detections']
        
        # Severity
        detections = severity_estimator.estimate_severity(image, detections)
        detections = severity_estimator.get_priority_ranking(detections)
        
        # Visualize
        annotated = BboxVisualizer.draw_detections(
            image, detections, color_by_severity=True
        )
        
        # Save
        output_path = os.path.join(output_dir, f'annotated_{image_path.stem}.jpg')
        cv2.imwrite(output_path, annotated)
        
        logger.info(f"  -> Found {len(detections)} damages")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on road damage detection model'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model')
    parser.add_argument('--image', type=str,
                       help='Path to single image (or use --dir for directory)')
    parser.add_argument('--dir', type=str,
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='results/inference',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config/inference_config.yaml',
                       help='Path to inference configuration')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.dir:
        logger.error("Either --image or --dir must be specified")
        sys.exit(1)
    
    if args.image:
        infer_image(args.model, args.image, args.output, args.config)
    elif args.dir:
        infer_directory(args.model, args.dir, args.output)


if __name__ == '__main__':
    main()
