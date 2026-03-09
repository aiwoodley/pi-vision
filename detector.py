#!/usr/bin/env python3
"""
Object Detection module for Raspberry Pi Vision AI
TensorFlow Lite based detection
"""

import os
import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import TFLite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = True
    except ImportError:
        logger.warning("TensorFlow Lite not available. Using mock detector.")
        TFLITE_AVAILABLE = False


class ObjectDetector:
    """TensorFlow Lite object detector"""
    
    # COCO labels (80 common objects)
    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, config, models_config):
        """Initialize the object detector"""
        self.config = config
        self.models_config = models_config
        self.interpreter = None
        self.model_name = None
        self.labels = []
        
        # Detection settings
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_detections = config.get('max_detections', 10)
        self.enabled = True
        
        # Input/output tensors
        self.input_tensor = None
        self.output_tensors = []
        
        # Load default model
        default_model = config.get('default_model', 'ssd_mobilenet_v2')
        self.load_model(default_model)
        
    def load_model(self, model_name):
        """Load a specific model"""
        if model_name not in self.models_config:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        model_info = self.models_config[model_name]
        model_path = Path(__file__).parent / model_info['file']
        labels_path = Path(__file__).parent / model_info['labels']
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Using mock detector for testing")
            self._setup_mock_detector()
            return True
            
        # Load labels
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = self.COCO_LABELS
            
        # Load TensorFlow Lite model
        try:
            self.interpreter = tflite.Interpreter(str(model_path))
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_name = model_name
            logger.info(f"Loaded model: {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._setup_mock_detector()
            return True
            
    def _setup_mock_detector(self):
        """Setup mock detector for testing without models"""
        self.interpreter = None
        logger.info("Using mock detector")
        
    def detect(self, frame):
        """Run object detection on a frame"""
        if not self.enabled:
            return []
            
        if self.interpreter is None:
            # Use mock detection for testing
            return self._mock_detect(frame)
            
        try:
            # Preprocess image
            input_shape = self.input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Resize and normalize
            img = cv2.resize(frame, (width, height))
            img = np.expand_dims(img, axis=0)
            img = (img.astype(np.float32) - 127.5) / 127.5
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            
            # Parse outputs (format varies by model)
            detections = self._parse_outputs()
            
            # Filter by confidence
            detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
            
            # Limit detections
            return detections[:self.max_detections]
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
            
    def _parse_outputs(self):
        """Parse model outputs into detection results"""
        # This varies by model - implementing SSD MobileNet V2 format
        try:
            # Get output tensors
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index']))
            
            detections = []
            for i in range(num_detections):
                if scores[0][i] < self.confidence_threshold:
                    continue
                    
                y1, x1, y2, x2 = boxes[0][i]
                
                detection = {
                    'class': self.labels[int(classes[0][i])] if int(classes[0][i]) < len(self.labels) else 'unknown',
                    'class_id': int(classes[0][i]),
                    'confidence': float(scores[0][i]),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }
                detections.append(detection)
                
            return detections
            
        except Exception as e:
            logger.error(f"Error parsing outputs: {e}")
            return []
            
    def _mock_detect(self, frame):
        """Generate mock detections for testing"""
        height, width = frame.shape[:2]
        
        # Return some dummy detections
        return [
            {
                'class': 'person',
                'class_id': 0,
                'confidence': 0.95,
                'bbox': {'x1': 0.2, 'y1': 0.3, 'x2': 0.5, 'y2': 0.9}
            },
            {
                'class': 'cup',
                'class_id': 42,
                'confidence': 0.78,
                'bbox': {'x1': 0.6, 'y1': 0.5, 'x2': 0.7, 'y2': 0.7}
            }
        ]
        
    def draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        if not detections:
            return frame
            
        height, width = frame.shape[:2]
        output = frame.copy()
        
        for det in detections:
            # Get bounding box coordinates
            x1 = int(det['bbox']['x1'] * width)
            y1 = int(det['bbox']['y1'] * height)
            x2 = int(det['bbox']['x2'] * width)
            y2 = int(det['bbox']['y2'] * height)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label background
            cv2.rectangle(output, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                       
        return output
        
    def next_model(self):
        """Switch to next available model"""
        model_names = list(self.models_config.keys())
        if not model_names:
            return
            
        current_idx = model_names.index(self.model_name) if self.model_name in model_names else -1
        next_idx = (current_idx + 1) % len(model_names)
        self.load_model(model_names[next_idx])
        
    def previous_model(self):
        """Switch to previous available model"""
        model_names = list(self.models_config.keys())
        if not model_names:
            return
            
        current_idx = model_names.index(self.model_name) if self.model_name in model_names else 0
        prev_idx = (current_idx - 1) % len(model_names)
        self.load_model(model_names[prev_idx])
