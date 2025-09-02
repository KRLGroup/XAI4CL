#!/bin/bash

# Build the Docker image
IMAGE_NAME="xai4cl-env"

echo "🔧 Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

echo "✅ Build complete."