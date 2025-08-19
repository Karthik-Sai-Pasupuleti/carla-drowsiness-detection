# Use official Python 3.12 Ubuntu image
FROM ubuntu:24.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv

# Upgrade pip inside venv
RUN /opt/venv/bin/python -m pip install --upgrade pip

# Install requirements inside venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Install CARLA wheel inside venv
COPY carla-0.10.0-cp312-cp312-linux_x86_64.whl .
RUN /opt/venv/bin/pip install --no-cache-dir ./carla-0.10.0-cp312-cp312-linux_x86_64.whl

# Copy project files
COPY . .

# Use the venv Python as default
ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "main.py"]
