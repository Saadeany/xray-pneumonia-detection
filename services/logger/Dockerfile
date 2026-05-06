FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# We run audit.py directly instead of uvicorn because it's a background loop, not a web server!
CMD ["python", "-u", "audit.py"]