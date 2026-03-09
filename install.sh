#!/bin/bash
# Raspberry Pi Vision AI - Installation Script
# Tested on Raspberry Pi 4 with Raspberry Pi OS Bookworm (64-bit)
#
# Run this on your Raspberry Pi:
#   sudo bash install.sh

set -e

echo "========================================"
echo "Raspberry Pi Vision AI - Installer"
echo "  (Raspberry Pi 4 / Bookworm)"
echo "========================================"

# ------------------------------------------------------------------
# Detect the real (non-root) user who invoked sudo
# ------------------------------------------------------------------
if [ -n "$SUDO_USER" ]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$(whoami)"
fi
TARGET_HOME=$(eval echo "~$TARGET_USER")
PROJECT_DIR="${TARGET_HOME}/projects/pi-vision"

echo ""
echo "Installing for user: ${TARGET_USER}"
echo "Project directory:   ${PROJECT_DIR}"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo bash $0"
    exit 1
fi

echo ""
echo "[1/7] Updating system..."
apt update && apt upgrade -y

echo ""
echo "[2/7] Installing system dependencies..."
apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    python3-libcamera \
    python3-tk \
    python3-pil \
    python3-pil.imagetk \
    libatlas-base-dev \
    libcap-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

echo ""
echo "[3/7] Verifying camera with libcamera..."
# On Bookworm the legacy camera stack is removed; libcamera is the default.
# Ensure the camera overlay is enabled in /boot/firmware/config.txt.
BOOT_CONFIG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CONFIG" ]; then
    BOOT_CONFIG="/boot/config.txt"   # Fallback for Bullseye
fi

# Make sure the camera dtoverlay is present
if ! grep -q "^dtoverlay=imx219\|^camera_auto_detect=1" "$BOOT_CONFIG" 2>/dev/null; then
    echo "camera_auto_detect=1" >> "$BOOT_CONFIG"
    echo "  → Added camera_auto_detect=1 to ${BOOT_CONFIG}"
fi

# Allocate GPU memory for camera + TFLite
if ! grep -q "^gpu_mem=" "$BOOT_CONFIG" 2>/dev/null; then
    echo "gpu_mem=128" >> "$BOOT_CONFIG"
    echo "  → Set gpu_mem=128 in ${BOOT_CONFIG}"
fi

echo ""
echo "[4/7] Setting up project directory..."
mkdir -p "${PROJECT_DIR}"
# Copy project files if this script is inside the repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/main.py" ]; then
    cp -r "${SCRIPT_DIR}"/* "${PROJECT_DIR}/"
    chown -R "${TARGET_USER}:${TARGET_USER}" "${PROJECT_DIR}"
fi

echo ""
echo "[5/7] Installing Python dependencies..."
cd "${PROJECT_DIR}"
sudo -u "${TARGET_USER}" pip3 install --break-system-packages -r requirements.txt || \
    sudo -u "${TARGET_USER}" pip3 install -r requirements.txt

echo ""
echo "[6/7] Downloading ML models..."
mkdir -p "${PROJECT_DIR}/models"
if [ -f "${PROJECT_DIR}/models/download_models.sh" ]; then
    cd "${PROJECT_DIR}/models"
    sudo -u "${TARGET_USER}" bash download_models.sh
else
    echo "  → models/download_models.sh not found — skipping model download."
    echo "    Place your .tflite models in ${PROJECT_DIR}/models/ manually."
fi

echo ""
echo "[7/7] Setting up auto-start (optional)..."
AUTOSTART_DIR="${TARGET_HOME}/.config/autostart"
mkdir -p "${AUTOSTART_DIR}"
cat > "${AUTOSTART_DIR}/pi-vision.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Pi Vision AI
Exec=python3 ${PROJECT_DIR}/main.py
Terminal=false
EOF
chown -R "${TARGET_USER}:${TARGET_USER}" "${AUTOSTART_DIR}"

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "IMPORTANT: Reboot to activate camera changes:"
echo "  sudo reboot"
echo ""
echo "To run manually after reboot:"
echo "  cd ${PROJECT_DIR}"
echo "  python3 main.py"
echo ""
echo "To verify your camera before running:"
echo "  libcamera-hello --timeout 5000"
echo ""
echo "Controls:"
echo "  Tap: Toggle detection"
echo "  Swipe L/R: Change model"
echo "  Swipe Up: Settings"
echo "  Swipe Down: Capture image"
echo "  ESC: Quit"
echo ""
