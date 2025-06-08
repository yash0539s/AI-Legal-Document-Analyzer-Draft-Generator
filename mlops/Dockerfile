FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies with extended timeout
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variable to include /app in the Python path
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

