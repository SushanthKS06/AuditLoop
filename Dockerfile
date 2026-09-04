FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data, metrics, and runtime directories
RUN mkdir -p data metrics runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AUDIT_DB_PATH=/app/runtime/audit_trail.db
ENV RESULTS_PATH=/app/runtime/results.json
ENV METRICS_PATH=/app/runtime/metrics_report.json

# Default command runs the pipeline with sample data
CMD ["python", "run_pipeline.py", "--force-disagreement"]
