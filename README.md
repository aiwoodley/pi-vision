# Raspberry Pi Vision AI - Touchscreen Object Detector

## Project Overview

A Raspberry Pi 4–powered intelligent vision system with touchscreen display that uses machine learning to detect and identify objects in real-time using the camera feed.

**What it does:**
- Streams camera feed to touchscreen display
- Runs real-time object detection using TensorFlow Lite
- Displays detection results with confidence scores
- Touch controls for switching models, adjusting sensitivity, and capturing images

**Target Users:**
- Learning AI/ML on edge devices
- Home automation integration
- Security and monitoring
- Educational demonstrations

---

## Bill of Materials

### Required Hardware
| Item | Approximate Cost | Notes |
|------|------------------|-------|
| Raspberry Pi 4 (4GB+) | $55–65 | 4GB minimum, 8GB recommended |
| Raspberry Pi 7" Touchscreen | $60–80 | Official DSI display |
| Raspberry Pi Camera Module v2 or v3 | $25–35 | CSI ribbon cable connection |
| 32GB+ microSD card | $10–15 | Class 10 / A2 recommended |
| Power supply (5V 3A USB-C) | $10–15 | Official Pi supply recommended |
| Optional: Case with fan | $15–30 | Keeps temps down during inference |

### Total Estimated Cost: $130–210

---

## Project Structure

```
pi-vision/
├── README.md
├── requirements.txt
├── config.json
├── install.sh
├── main.py
├── camera.py          ← picamera2 + OpenCV backends
├── detector.py        ← TFLite inference (quantised model support)
├── ui.py
├── test_mode.py
├── models/
│   ├── download_models.sh
│   ├── ssd_mobilenet_v2_coco_quant_postprocess.tflite
│   ├── efficientdet_lite0.tflite
│   └── coco_labels.txt
├── assets/
│   └── icons/
└── data/
    └── captures/
```

---

## Setup Instructions (Raspberry Pi 4 + Bookworm)

### 1. Flash Raspberry Pi OS

Download **Raspberry Pi OS (64-bit) with desktop** using the Raspberry Pi Imager.
Enable SSH and configure WiFi in the imager's settings before flashing.

> **Important:** Use the 64-bit (aarch64) image. TFLite and picamera2 work
> best on 64-bit Bookworm.

### 2. Initial Setup

```bash
# SSH into your Pi (replace <username> with the user you configured)
ssh <username>@raspberrypi.local

# Update system
sudo apt update && sudo apt upgrade -y
```

### 3. Verify the Camera

On Bookworm the **libcamera** stack is used by default (the legacy `raspicam`
interface is removed). Test with:

```bash
# Quick camera test — you should see a 5-second preview
libcamera-hello --timeout 5000

# If you see an error, check the ribbon cable and ensure the camera
# overlay is enabled in /boot/firmware/config.txt:
#   camera_auto_detect=1
```

### 4. Automated Install

```bash
cd ~
mkdir -p projects && cd projects
git clone <this-repo> pi-vision
cd pi-vision

# Run the installer (handles packages, picamera2, models, autostart)
sudo bash install.sh

# Reboot to apply camera / GPU memory changes
sudo reboot
```

### 5. Manual Install (alternative)

```bash
# System packages (includes picamera2 & OpenCV from apt)
sudo apt install -y \
    python3-pip python3-opencv python3-picamera2 python3-libcamera \
    python3-tk python3-pil python3-pil.imagetk libatlas-base-dev

# Python packages
cd ~/projects/pi-vision
pip3 install --break-system-packages -r requirements.txt

# Download models
cd models && bash download_models.sh
```

### 6. Run the Application

```bash
cd ~/projects/pi-vision
python3 main.py
```

---

## Configuration

Edit `config.json` to customise behaviour. Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `camera.backend` | `"auto"` | `"auto"`, `"picamera2"`, or `"opencv"` |
| `camera.width` / `height` | 640 × 480 | Lower for faster inference |
| `camera.fps` | 24 | 24 is comfortable on Pi 4 |
| `detection.confidence_threshold` | 0.5 | 0.0 – 1.0 |
| `performance.gpu_memory_mb` | 128 | GPU split in MB (set in `/boot/firmware/config.txt`) |

---

## Usage Guide

### Touch Controls
- **Tap anywhere**: Toggle detection on/off
- **Swipe left/right**: Switch between detection models
- **Swipe up**: Open settings menu
- **Swipe down**: Capture and save image
- **ESC**: Quit

### Test Mode (no hardware)

```bash
python3 test_mode.py
```

Runs with a mock camera so you can test the UI and detection pipeline on
any machine.

---

## Technical Details

### ML Models
1. **SSD MobileNet V2 (quantised)** — ~120 ms inference on Pi 4
2. **EfficientDet-Lite0** — ~200 ms inference, better accuracy

Both are uint8-quantised TFLite models optimised for ARM Cortex-A72.

### Performance Targets (Pi 4, 4GB)
- 8–15 FPS with SSD MobileNet V2 at 640×480
- < 500 MB RAM usage
- Runs comfortably at the default 1.5 GHz clock

### Camera Stack
This project uses **picamera2** (the official Python interface for libcamera)
as the primary camera backend. If picamera2 is not available (e.g. when using
a USB webcam), it falls back automatically to OpenCV's V4L2 backend.

---

## Troubleshooting

### Camera not detected
```bash
# Test with libcamera directly
libcamera-hello --timeout 3000

# Check the ribbon cable is seated firmly on both ends
# Ensure /boot/firmware/config.txt contains:
#   camera_auto_detect=1

# Reboot after changes
sudo reboot
```

### "picamera2 not found"
```bash
sudo apt install -y python3-picamera2 python3-libcamera
```

### Low FPS
- Set `camera.width` / `height` to 320×240 in `config.json`
- Use the SSD MobileNet V2 model (lighter)
- Overclock cautiously via `/boot/firmware/config.txt`

### Out of memory
- Use quantised (uint8) models only
- Set `gpu_mem=128` in `/boot/firmware/config.txt`
- Close the desktop browser and other apps

---

## Next Steps / Extensions

1. **Add voice announcements** — Text-to-speech for detections
2. **Integrate with Home Assistant** — MQTT notifications
3. **Custom model training** — Train on specific objects with TFLite Model Maker
4. **Time-lapse recording** — Capture detection events
5. **Edge TPU acceleration** — Add Google Coral USB accelerator (~10× faster)

---

## License

MIT License — Modify and use freely for your projects.
