#!/bin/bash
# Raspberry Pi Vision AI - Installation Script
# Run this on your Raspberry Pi

set -e

echo "========================================"
echo "Raspberry Pi Vision AI - Installer"
echo "========================================"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo $0"
    exit 1
fi

echo ""
echo "[1/6] Updating system..."
apt update && apt upgrade -y

echo ""
echo "[2/6] Installing system dependencies..."
apt install -y python3-pip python3-venv libopencv-dev libatlas-base-dev libilmbase-dev libopenexr-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

echo ""
echo "[3/6] Enabling camera interface..."
raspi-config nonint do_camera 0

echo ""
echo "[4/6] Installing Python dependencies..."
cd /home/pi/projects/pi-vision
pip3 install -r requirements.txt

echo ""
echo "[5/6] Downloading ML models..."
cd models
bash download_models.sh

echo ""
echo "[6/6] Setting up auto-start..."
mkdir -p /home/pi/.config/autostart
cat > /home/pi/.config/autostart/pi-vision.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Pi Vision AI
Exec=python3 /home/pi/projects/pi-vision/main.py
Terminal=false
EOF

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "To run manually:"
echo "  cd ~/projects/pi-vision"
echo "  python3 main.py"
echo ""
echo "The app will start automatically on next reboot."
echo ""
echo "Controls:"
echo "  Tap: Toggle detection"
echo "  Swipe L/R: Change model"
echo "  Swipe Up: Settings"
echo "  Swipe Down: Capture image"
echo "  ESC: Quit"
echo ""
