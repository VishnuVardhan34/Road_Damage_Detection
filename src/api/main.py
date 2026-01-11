"""
FastAPI application for road damage detection inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import logging
from typing import List, Optional
import yaml
import os

logger = logging.getLogger(__name__)


class APIConfig:
    """API configuration."""
    
    def __init__(self, config_path: str = 'config/inference_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)


class RoadDamageAPI:
    """Road Damage Detection API."""
    
    def __init__(self, model_path: str, config_path: str = 'config/inference_config.yaml'):
        """
        Initialize API.
        
        Args:
            model_path: Path to YOLO model
            config_path: Path to inference configuration
        """
        from src.models.yolo_detector import YOLODetector
        from src.models.severity_estimator import SeverityEstimator
        
        self.config = APIConfig(config_path)
        self.detector = YOLODetector(
            model_path,
            device='cuda',
            conf_threshold=self.config.config['post_processing']['confidence_threshold'],
            nms_threshold=self.config.config['post_processing']['nms_threshold']
        )
        self.severity_estimator = SeverityEstimator()
        
        logger.info("API initialized successfully")
    
    def process_image(self, image_data: bytes) -> dict:
        """
        Process uploaded image.
        
        Args:
            image_data: Image file bytes
            
        Returns:
            Detection results with severity
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Run detection
            result = self.detector.detect(image)
            detections = result['detections']
            
            # Estimate severity
            detections_with_severity = self.severity_estimator.estimate_severity(
                image, detections
            )
            
            # Generate report
            report = self.severity_estimator.generate_severity_report(
                detections_with_severity
            )
            
            return {
                'status': 'success',
                'num_detections': len(detections_with_severity),
                'detections': [
                    {
                        'class_name': det['class_name'],
                        'confidence': float(det['confidence']),
                        'severity': float(det.get('severity', 0)),
                        'severity_category': det.get('severity_category', 'Unknown'),
                        'bbox': det['bbox']
                    }
                    for det in detections_with_severity
                ],
                'severity_report': report
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise


def create_app(model_path: str, config_path: str = 'config/inference_config.yaml') -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to YOLO model
        config_path: Path to configuration
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Road Damage Detection API",
        description="AI-based road damage and severity detection system",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize API
    api = RoadDamageAPI(model_path, config_path)
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "healthy", "service": "Road Damage Detection API"}
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok"}
    
    @app.post("/detect")
    async def detect(file: UploadFile = File(...)):
        """
        Detect road damages in uploaded image.
        
        Args:
            file: Image file (JPG, PNG, etc.)
            
        Returns:
            Detection results with severity scores
        """
        # Check file size
        max_size_mb = api.config.config['api']['max_image_size_mb']
        contents = await file.read()
        
        if len(contents) > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {max_size_mb}MB)"
            )
        
        # Check file type
        supported_formats = api.config.config['api']['supported_formats']
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {supported_formats}"
            )
        
        # Process image
        try:
            result = api.process_image(contents)
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {str(e)}"
            )
    
    @app.post("/detect_batch")
    async def detect_batch(files: List[UploadFile] = File(...)):
        """
        Detect damages in multiple images.
        
        Args:
            files: List of image files
            
        Returns:
            List of detection results
        """
        results = []
        
        for file in files:
            contents = await file.read()
            try:
                result = api.process_image(contents)
                result['filename'] = file.filename
                results.append(result)
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {'results': results}
    
    @app.get("/info")
    async def info():
        """Get API information."""
        return {
            'name': 'Road Damage Detection API',
            'version': '1.0.0',
            'model': 'YOLOv8L',
            'classes': [
                'Crack', 'Pothole', 'Rutting', 'Patch', 'Lane_Wear', 'Manhole'
            ],
            'confidence_threshold': api.config.config['post_processing']['confidence_threshold'],
            'nms_threshold': api.config.config['post_processing']['nms_threshold']
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Load model path from environment or default
    model_path = os.getenv('MODEL_PATH', 'checkpoints/best_model.pt')
    
    app = create_app(model_path)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
