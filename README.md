# Raspberry Pi Vision AI - Touchscreen Object Detector

## Project Overview

A Raspberry Pi-powered intelligent vision system with touchscreen display that uses machine learning to detect and identify objects in real-time using the camera feed.

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
| Raspberry Pi 4 or 5 (4GB+) | $55-80 | 4GB minimum, 8GB recommended |
| Raspberry Pi 7" Touchscreen | $60-80 | Official display |
| Raspberry Pi Camera v2 | $25-30 | Or Camera Module 3 |
| 32GB+ microSD card | $15 | Class 10 recommended |
| Power supply (5V 3A USB-C) | $15 | Official Pi supply recommended |
| Optional: Case with fan | $20-30 | For thermal management |

### Total Estimated Cost: $130-220

---

## Project Structure

```
pi-vision/
├── README.md
├── requirements.txt
├── config.json
├── main.py
├── detector.py
├── ui.py
├── camera.py
├── models/
│   └── download_models.sh
├── assets/
│   └── icons/
└── data/
    └── captures/
```

---

## Setup Instructions

### 1. Flash Raspberry Pi OS

```bash
# Download Raspberry Pi Imager from raspberrypi.com
# Flash Raspberry Pi OS (64-bit) with desktop environment
# Enable SSH and configure WiFi in the imager settings
```

### 2. Initial Setup

```bash
# SSH into your Pi
ssh pi@raspberrypi.local
# password: raspberry

# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera and touchscreen
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → I2C → Enable
# Display Options → DSI Display → Enable
```

### 3. Install Dependencies

```bash
# Install system packages
sudo apt install -y python3-pip libopencv-dev libatlas-base-dev

# Clone and setup project
cd ~
mkdir -p projects && cd projects
git clone <this-repo> pi-vision
cd pi-vision

# Install Python dependencies
pip3 install -r requirements.txt
```

### 4. Download ML Models

```bash
cd models
bash download_models.sh
```

### 5. Run the Application

```bash
# From the project directory
cd ~/projects/pi-vision
python3 main.py
```

---

## Usage Guide

### Touch Controls
- **Tap anywhere**: Toggle detection on/off
- **Swipe left/right**: Switch between detection models
- **Tap detection box**: Show detailed info
- **Swipe up**: Open settings menu
- **Swipe down**: Capture and save image

### Settings Menu
- Detection threshold slider
- Model selection
- Display options (fullscreen, overlay opacity)
- Camera settings (brightness, contrast)

---

## Technical Details

### ML Models Used
1. **SSD MobileNet V2** - Fast, good accuracy, ~100ms inference
2. **EfficientDet-Lite0** - Better accuracy, ~150ms inference
3. **YOLOv8-tiny** - Best balance, ~50ms inference

### Performance Targets
- 15-30 FPS on Raspberry Pi 4
- < 500MB RAM usage
- < 2W power consumption (display on)

### Supported Detections
Uses COCO dataset (80 common objects):
- People, vehicles, animals, household items
- Food, furniture, electronics, sports equipment

---

## Troubleshooting

### Camera not detected
```bash
# Check camera connection
vcgencmd get_camera

# Enable camera if disabled
sudo raspi-config
```

### Low FPS
- Reduce display resolution
- Use lighter model (YOLOv8-tiny)
- Overclock Pi (edit /boot/config.txt)

### Out of memory
- Use smaller model
- Reduce camera resolution
- Close other applications

---

## Next Steps / Extensions

1. **Add voice announcements** - Text-to-speech for detections
2. **Integrate with home assistant** - MQTT notifications
3. **Custom model training** - Train on specific objects
4. **Time-lapse recording** - Capture detection events
5. **Edge TPU acceleration** - Add Google Coral USB accelerator

---

## License

MIT License - Modify and use freely for your projects.
