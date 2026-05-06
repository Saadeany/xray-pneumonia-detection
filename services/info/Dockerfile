# ── 1. Use an ultra-lightweight Python base image
FROM python:3.10-slim-bookworm

# ── 2. Set the working directory
WORKDIR /app

# ── 3. Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Copy the application code
COPY metadata.py .

# ── 5. Expose the internal port
EXPOSE 8000

# ── 6. Boot the server 
# CRITICAL: We use "metadata:app" because your Python file is named metadata.py!
CMD ["uvicorn", "metadata:app", "--host", "0.0.0.0", "--port", "8000"]