FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads templates

# Make the uploads directory writable
RUN chmod 777 static/uploads

# Default port (will be overridden by Render)
ENV PORT=8000

# Command to run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT 