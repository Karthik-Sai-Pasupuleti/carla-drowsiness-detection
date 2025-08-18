# Use official Python 3.12 slim image (Debian-based)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CARLA wheel and install it
COPY carla-0.10.0-cp312-cp312-linux_x86_64.whl .
RUN pip install --no-cache-dir ./carla-0.10.0-cp312-cp312-linux_x86_64.whl


# Copy project files
COPY . .

# Run main application
CMD ["python", "main.py"]
