# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY getTensorFromFen.py .
COPY parser.py .
COPY load_dataset.py .
COPY chess_cnn.py .
COPY train.py .

# Create directory for data and models
RUN mkdir -p /app/data /app/models

# Default command (load dataset then run training)
CMD ["sh", "-c", "python load_dataset.py && python train.py"]
