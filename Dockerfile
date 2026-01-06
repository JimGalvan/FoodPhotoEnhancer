# Use Python 3.10.13 slim image based on Debian Bookworm
FROM python:3.10.13-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints photos temp

# Expose port 8000
EXPOSE 8000

# Run gunicorn with configuration
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--config", "gunicorn.conf.py"]
