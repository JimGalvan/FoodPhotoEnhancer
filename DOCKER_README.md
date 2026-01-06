# Docker Deployment Guide for FoodPhotoEnhancer

This guide explains how to build and deploy the FoodPhotoEnhancer inference service using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 2: Using Docker Commands

```bash
# Build the image
docker build -t foodphotoenhancer:latest .

# Run the container
docker run -d \
  -p 8000:8000 \
  --name foodphotoenhancer \
  -v photos-data:/app/photos \
  -v checkpoints-data:/app/checkpoints \
  -v temp-data:/app/temp \
  foodphotoenhancer:latest

# View logs
docker logs -f foodphotoenhancer

# Stop the container
docker stop foodphotoenhancer
docker rm foodphotoenhancer
```

## Accessing the Service

Once running, the service will be available at:
- **Base URL**: `http://localhost:8000`
- **Upload Endpoint**: `http://localhost:8000/upload_photo` (or your configured endpoint)

## Environment Variables

If you need to configure environment variables, create a `.env` file in the project root:

```env
DJANGO_SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
# Add other environment variables as needed
```

Then update docker-compose.yml to use the env file:

```yaml
services:
  web:
    env_file:
      - .env
```

## Production Deployment

For production deployment:

1. **Build the image**:
   ```bash
   docker build -t foodphotoenhancer:production .
   ```

2. **Push to a registry** (if using a container registry):
   ```bash
   docker tag foodphotoenhancer:production your-registry/foodphotoenhancer:latest
   docker push your-registry/foodphotoenhancer:latest
   ```

3. **Deploy to your platform** (Railway, AWS, GCP, etc.)

## Troubleshooting

### Container fails to start
```bash
# Check logs
docker logs foodphotoenhancer

# Check if port 8000 is already in use
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows
```

### Out of memory
The ML models can be memory-intensive. Increase Docker's memory limit:
- Docker Desktop: Settings → Resources → Memory (recommend at least 4GB)

### Models not loading
Ensure the checkpoints directory is properly mounted and contains the required model files.

## Performance Optimization

For better performance in production:

1. **Adjust Gunicorn workers** in `gunicorn.conf.py`:
   ```python
   workers = 2  # Or more, based on CPU cores
   ```

2. **Use multi-stage build** for smaller images (optional)

3. **Enable model preloading** to reduce cold start time

## Monitoring

To monitor the running container:

```bash
# View resource usage
docker stats foodphotoenhancer

# Execute commands inside container
docker exec -it foodphotoenhancer bash
```

## Cleaning Up

```bash
# Remove container
docker stop foodphotoenhancer && docker rm foodphotoenhancer

# Remove image
docker rmi foodphotoenhancer:latest

# Remove volumes (WARNING: This deletes all data)
docker volume rm photos-data checkpoints-data temp-data
```
