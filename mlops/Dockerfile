# Use official slim Python image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /lexidraft_pro

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies with retries & timeout
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Copy the full project
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
