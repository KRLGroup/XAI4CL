#!/bin/bash

# Run the Docker container with GPU support
IMAGE_NAME="xai4cl-env"
CONTAINER_NAME="xai4cl-container"

echo "🚀 Running Docker container: $CONTAINER_NAME ..."

docker run --gpus all -it --rm \
    --name $CONTAINER_NAME \
    -v $(pwd):/workspace \
    $IMAGE_NAME

echo "🛑 Container stopped."