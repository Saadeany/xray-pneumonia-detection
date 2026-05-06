# ── 1. Use a lightweight Python base image
FROM python:3.10-slim-bookworm

# ── 2. Set the working directory inside the container
WORKDIR /app

# ── 3. The Senior Trick: Install PyTorch CPU-only version FIRST.
# This prevents Docker from downloading 4GB of useless GPU drivers for inference.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── 4. Copy the requirements file and install the web dependencies (FastAPI, Uvicorn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 5. Copy the rest of the application code (including app.py and xray_classifier.pth)
COPY . .

# ── 6. Expose the port the FastAPI server will run on
EXPOSE 8000

# ── 7. Boot up the server.
# CRITICAL: We use "app:app" here because you named the Python file app.py!
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]