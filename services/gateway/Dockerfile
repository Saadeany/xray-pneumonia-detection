# ── 1. Use an ultra-lightweight Python base image
FROM python:3.10-slim-bookworm

# ── 2. Set the working directory
WORKDIR /app

# ── 3. Copy requirements and install them securely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Copy the application code
COPY main.py .

# ── 5. Expose the port to the outside world
EXPOSE 8000

# ── 6. Boot the Gateway server
# CRITICAL: "main:app" because your Python file is main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]