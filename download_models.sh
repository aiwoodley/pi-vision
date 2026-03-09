#!/bin/bash
# Download TensorFlow Lite models for Raspberry Pi Vision AI

set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

echo "Downloading TensorFlow Lite models..."

# SSD MobileNet V2 (COCO)
# This is a quantized model optimized for edge devices
echo "Downloading SSD MobileNet V2..."
curl -L -o "$MODELS_DIR/ssd_mobilenet_v2_coco_quant_postprocess.tflite" \
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v2_1.0_quant_postprocess.tflite"

# EfficientDet-Lite0
echo "Downloading EfficientDet-Lite0..."
curl -L -o "$MODELS_DIR/efficientdet_lite0.tflite" \
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/efficientdet_lite0_v1.0.tar.gz"

# Extract EfficientDet if downloaded as tarball
if file "$MODELS_DIR/efficientdet_lite0.tflite" | grep -q "gzip"; then
    mv "$MODELS_DIR/efficientdet_lite0.tflite" "$MODELS_DIR/efficientdet_lite0.tar.gz"
    tar -xzf "$MODELS_DIR/efficientdet_lite0.tar.gz" -C "$MODELS_DIR"
    rm "$MODELS_DIR/efficientdet_lite0.tar.gz"
fi

echo "Models downloaded successfully!"
echo "Files in $MODELS_DIR:"
ls -la "$MODELS_DIR"
