FROM python:3.11-slim

# Force Python to flush stdout/stderr immediately (visible logs in Docker)
ENV PYTHONUNBUFFERED=1
# It's a free rate-limit key, not a secret credential
ENV API_KEY = cvmNVUH4Db7ea9Bj2Al5LFaud1qMyXK3Hc0eIDdA

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application folders
COPY app/ ./app/
COPY data/ ./data/
COPY eda/ ./eda/
COPY model/ ./model/

# Make all folders importable across each other
ENV PYTHONPATH=/app

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
