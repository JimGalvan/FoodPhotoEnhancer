#!/bin/bash

# Build script for FoodPhotoEnhancer Docker image

set -e

echo "Building FoodPhotoEnhancer Docker image..."
docker build -t foodphotoenhancer:latest .

echo ""
echo "Build complete! Run the service with:"
echo "  docker-compose up -d"
echo "or"
echo "  docker run -d -p 8000:8000 --name foodphotoenhancer foodphotoenhancer:latest"
