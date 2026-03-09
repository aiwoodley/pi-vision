#!/usr/bin/env python3
"""
Camera module for Raspberry Pi Vision AI
Handles camera capture and configuration
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Camera:
    """Camera controller for Raspberry Pi"""
    
    def __init__(self, config):
        """Initialize camera with configuration"""
        self.config = config
        self.cap = None
        self.frame = None
        
        # Camera parameters
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.rotation = config.get('rotation', 0)
        
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Initialize the camera connection"""
        # Try different camera backends
        backends = [
            (cv2.CAP_V4L2, 'V4L2'),
            (cv2.CAP_ANY, 'ANY')
        ]
        
        for backend, name in backends:
            logger.info(f"Trying camera backend: {name}")
            self.cap = cv2.VideoCapture(0, backend)
            
            if self.cap.isOpened():
                logger.info(f"Camera opened with {name} backend")
                break
                
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Apply initial settings
        if 'brightness' in self.config:
            self.set_brightness(self.config['brightness'])
        if 'contrast' in self.config:
            self.set_contrast(self.config['contrast'])
            
        logger.info(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
        
    def read_frame(self):
        """Read a frame from the camera"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Apply rotation if needed
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        return frame
        
    def set_brightness(self, value):
        """Set camera brightness (0-100)"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value / 100.0)
            
    def set_contrast(self, value):
        """Set camera contrast (0-100)"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_CONTRAST, value / 100.0)
            
    def set_exposure(self, value):
        """Set camera exposure (-7 to -1, or auto)"""
        if self.cap and self.cap.isOpened():
            if value == 'auto':
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = auto
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual
                self.cap.set(cv2.CAP_PROP_EXPOSURE, max(-7, min(-1, value)))
                
    def save_frame(self, path, frame=None):
        """Save a frame to disk"""
        if frame is None:
            frame = self.frame
            
        if frame is not None:
            cv2.imwrite(str(path), frame)
            
    def cleanup(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            logger.info("Camera released")


class MockCamera:
    """Mock camera for testing without hardware"""
    
    def __init__(self, config):
        self.config = config
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.brightness = config.get('brightness', 50)
        self.contrast = config.get('contrast', 50)
        
    def read_frame(self):
        """Generate a test pattern frame"""
        # Create gradient test pattern
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add gradient
        for y in range(self.height):
            for x in range(self.width):
                frame[y, x] = [
                    int(255 * x / self.width),
                    int(255 * y / self.height),
                    128
                ]
                
        # Add some shapes to detect
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(frame, (400, 240), 50, (0, 255, 0), -1)
        
        return frame
        
    def set_brightness(self, value):
        self.brightness = value
        
    def set_contrast(self, value):
        self.contrast = value
        
    def cleanup(self):
        pass
