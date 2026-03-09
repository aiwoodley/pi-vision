#!/usr/bin/env python3
"""
Test version that runs without Raspberry Pi hardware
Uses mock camera for testing the full application
"""

import os
import sys
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from camera import MockCamera
from detector import ObjectDetector
from ui import TouchUI

# Use mock camera instead of real camera
from camera import Camera as RealCamera

# Monkey-patch to use mock camera
import main as main_module

# Override the Camera import in main.py
original_init = main_module.Camera

class TestCamera:
    """Test camera that generates fake frames"""
    
    def __init__(self, config):
        self.config = config
        self.mock = MockCamera(config)
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.brightness = config.get('brightness', 50)
        self.contrast = config.get('contrast', 50)
        
    def read_frame(self):
        return self.mock.read_frame()
        
    def set_brightness(self, value):
        self.brightness = value
        self.mock.set_brightness(value)
        
    def set_contrast(self, value):
        self.contrast = value
        self.mock.set_contrast(value)
        
    def cleanup(self):
        pass

# Replace Camera in main module
main_module.Camera = TestCamera


def main():
    """Run the test version"""
    print("=" * 50)
    print("Raspberry Pi Vision AI - Test Mode")
    print("Running with mock camera (no hardware needed)")
    print("=" * 50)
    print()
    print("Controls:")
    print("  Click: Toggle detection")
    print("  Swipe L/R: Change model")
    print("  ESC: Quit")
    print()
    print("Starting in 3 seconds...")
    print()
    
    import time
    time.sleep(3)
    
    # Import and run main
    main_module.main()


if __name__ == "__main__":
    main()
