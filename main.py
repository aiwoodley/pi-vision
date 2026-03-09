#!/usr/bin/env python3
"""
Raspberry Pi Vision AI - Main Application
Touchscreen object detection with TensorFlow Lite
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from camera import Camera
from detector import ObjectDetector
from ui import TouchUI


class VisionAI:
    """Main application controller"""
    
    def __init__(self, config_path=None):
        """Initialize the Vision AI system"""
        # Load configuration
        if config_path is None:
            config_path = PROJECT_ROOT / "config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Starting {self.config['project']['name']} v{self.config['project']['version']}")
        
        # Initialize components
        self.camera = None
        self.detector = None
        self.ui = None
        self.running = False
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing camera...")
        self.camera = Camera(self.config['camera'])
        
        logger.info("Loading ML models...")
        self.detector = ObjectDetector(
            self.config['detection'],
            self.config['models']
        )
        
        logger.info("Starting UI...")
        self.ui = TouchUI(
            self.config['display'],
            self.config['detection'],
            on_model_change=self.handle_model_change,
            on_settings_change=self.handle_settings_change,
            on_capture=self.handle_capture
        )
        
        logger.info("Initialization complete!")
        
    def handle_model_change(self, model_name):
        """Handle model switching request"""
        logger.info(f"Switching to model: {model_name}")
        self.detector.load_model(model_name)
        
    def handle_settings_change(self, settings):
        """Handle settings changes"""
        logger.info(f"Updating settings: {settings}")
        if 'threshold' in settings:
            self.detector.confidence_threshold = settings['threshold']
        if 'brightness' in settings:
            self.camera.set_brightness(settings['brightness'])
        if 'contrast' in settings:
            self.camera.set_contrast(settings['contrast'])
            
    def handle_capture(self, frame, detections):
        """Handle image capture"""
        capture_dir = PROJECT_ROOT / self.config['capture']['directory']
        capture_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw frame
        raw_path = capture_dir / f"capture_{timestamp}.jpg"
        self.camera.save_frame(raw_path)
        
        # Save annotated frame if enabled
        if self.config['capture']['save_annotated'] and detections:
            annotated = self.detector.draw_detections(frame, detections)
            annotated_path = capture_dir / f"annotated_{timestamp}.jpg"
            self.camera.save_frame(annotated_path, frame=annotated)
            
        logger.info(f"Captured: {raw_path}")
        
    def run(self):
        """Main run loop"""
        self.running = True
        logger.info("Starting main loop...")
        
        while self.running:
            try:
                # Capture frame
                frame = self.camera.read_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Update UI
                self.ui.update(frame, detections, self.fps)
                
                # Handle UI events
                event = self.ui.check_event()
                if event:
                    self.handle_ui_event(event)
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(0.5)
                
        self.cleanup()
        
    def handle_ui_event(self, event):
        """Process UI events"""
        event_type = event.get('type')
        
        if event_type == 'toggle':
            self.detector.enabled = not self.detector.enabled
            logger.info(f"Detection {'enabled' if self.detector.enabled else 'disabled'}")
            
        elif event_type == 'swipe_left':
            self.detector.next_model()
            
        elif event_type == 'swipe_right':
            self.detector.previous_model()
            
        elif event_type == 'swipe_up':
            self.ui.show_settings()
            
        elif event_type == 'swipe_down':
            frame = self.camera.read_frame()
            if frame is not None:
                detections = self.detector.detect(frame) if self.detector.enabled else []
                self.handle_capture(frame, detections)
                
        elif event_type == 'quit':
            self.running = False
            
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        if self.ui:
            self.ui.cleanup()
        if self.camera:
            self.camera.cleanup()
        logger.info("Shutdown complete!")


def main():
    """Entry point"""
    # Check for config path argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and run application
    app = VisionAI(config_path)
    
    try:
        app.initialize()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
