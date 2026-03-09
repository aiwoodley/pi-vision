#!/usr/bin/env python3
"""
Camera module for Raspberry Pi Vision AI
Handles camera capture and configuration

Supports:
  - picamera2 (libcamera stack, Pi 4 / Bookworm default)
  - OpenCV V4L2 fallback (USB cameras or legacy setups)
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import picamera2 (preferred on Pi 4 with Bookworm)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logger.info("picamera2 not available — will use OpenCV backend")


class Camera:
    """Camera controller for Raspberry Pi"""

    def __init__(self, config):
        """Initialize camera with configuration"""
        self.config = config
        self.cap = None          # OpenCV capture (fallback)
        self.picam2 = None       # picamera2 instance (preferred)
        self.backend = None      # 'picamera2' | 'opencv'
        self.frame = None

        # Camera parameters
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.rotation = config.get('rotation', 0)

        # Allow the user to force a backend via config
        preferred = config.get('backend', 'auto')  # 'auto', 'picamera2', 'opencv'
        self._initialize_camera(preferred)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_camera(self, preferred='auto'):
        """Try to open the camera using the best available backend."""

        if preferred in ('auto', 'picamera2') and PICAMERA2_AVAILABLE:
            try:
                self._init_picamera2()
                return
            except Exception as e:
                logger.warning(f"picamera2 init failed: {e}")

        # Fallback to OpenCV (USB cameras or legacy setups)
        if preferred in ('auto', 'opencv'):
            self._init_opencv()
            return

        raise RuntimeError(
            f"Requested backend '{preferred}' is not available. "
            "Install picamera2 or ensure a V4L2 camera is connected."
        )

    def _init_picamera2(self):
        """Initialise using the libcamera / picamera2 stack (Pi 4 default)."""
        logger.info("Initialising camera via picamera2 (libcamera)...")
        self.picam2 = Picamera2()

        cam_config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=2,
        )
        self.picam2.configure(cam_config)

        # Apply 180° rotation via libcamera transform when possible
        if self.rotation == 180:
            try:
                from libcamera import Transform
                cam_config["transform"] = Transform(hflip=True, vflip=True)
                self.picam2.configure(cam_config)
            except ImportError:
                logger.info("libcamera Transform not importable; 180° handled in software")

        self.picam2.start()
        self.backend = 'picamera2'
        logger.info(
            f"Camera initialised (picamera2): {self.width}x{self.height} @ {self.fps}fps"
        )

    def _init_opencv(self):
        """Initialise using OpenCV V4L2 (USB cameras / legacy setups)."""
        backends = [
            (cv2.CAP_V4L2, 'V4L2'),
            (cv2.CAP_ANY, 'ANY'),
        ]

        for backend, name in backends:
            logger.info(f"Trying camera backend: {name}")
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                logger.info(f"Camera opened with {name} backend")
                break

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Failed to open camera via OpenCV")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if 'brightness' in self.config:
            self.set_brightness(self.config['brightness'])
        if 'contrast' in self.config:
            self.set_contrast(self.config['contrast'])

        self.backend = 'opencv'
        logger.info(
            f"Camera initialised (OpenCV): {self.width}x{self.height} @ {self.fps}fps"
        )

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read_frame(self):
        """Read a frame from the camera (returns BGR numpy array)."""
        frame = None

        if self.backend == 'picamera2' and self.picam2:
            # picamera2 returns RGB; convert to BGR for OpenCV consistency
            rgb = self.picam2.capture_array()
            if rgb is not None:
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif self.backend == 'opencv' and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return None

        if frame is None:
            return None

        # Software rotation for 90° / 270° (not handled by libcamera transform)
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180 and self.backend != 'picamera2':
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------

    def set_brightness(self, value):
        """Set camera brightness (0-100)"""
        if self.backend == 'picamera2' and self.picam2:
            # Map 0-100 → libcamera Brightness range (-1.0 to 1.0)
            mapped = (value / 50.0) - 1.0
            self.picam2.set_controls({"Brightness": mapped})
        elif self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value / 100.0)

    def set_contrast(self, value):
        """Set camera contrast (0-100)"""
        if self.backend == 'picamera2' and self.picam2:
            # Map 0-100 → libcamera Contrast range (0.0 to 2.0)
            mapped = value / 50.0
            self.picam2.set_controls({"Contrast": mapped})
        elif self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_CONTRAST, value / 100.0)

    def set_exposure(self, value):
        """Set camera exposure"""
        if self.backend == 'picamera2' and self.picam2:
            if value == 'auto':
                self.picam2.set_controls({"AeEnable": True})
            else:
                # value in microseconds for libcamera
                self.picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": int(abs(value) * 10000),
                })
        elif self.cap and self.cap.isOpened():
            if value == 'auto':
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, max(-7, min(-1, value)))

    def save_frame(self, path, frame=None):
        """Save a frame to disk"""
        if frame is None:
            frame = self.frame
        if frame is not None:
            cv2.imwrite(str(path), frame)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release camera resources"""
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            logger.info("picamera2 released")
        if self.cap:
            self.cap.release()
            logger.info("OpenCV camera released")


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
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                frame[y, x] = [
                    int(255 * x / self.width),
                    int(255 * y / self.height),
                    128,
                ]

        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(frame, (400, 240), 50, (0, 255, 0), -1)

        return frame

    def set_brightness(self, value):
        self.brightness = value

    def set_contrast(self, value):
        self.contrast = value

    def cleanup(self):
        pass
