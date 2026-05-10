import os
import json
import pika
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="X-Ray Pneumonia Detection API",
    description="Medical AI Gateway — orchestrates classifier, severity, and compliance services.",
    version="1.0.0",
)

# ── 1. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Internal Microservice URLs (Docker Compose DNS)
CLASSIFIER_URL = "http://classifier:8000/predict"
SEVERITY_URL   = "http://severity:8000/evaluate"
INFO_URL       = "http://info:8000/metadata"
RABBITMQ_HOST  = os.getenv("RABBITMQ_HOST", "rabbitmq")

# ── 3. Ethical Disclaimer (required by spec)
DISCLAIMER = "Screening tool only. Not a substitute for professional clinical diagnosis. Always consult a licensed radiologist."


def send_to_audit_log(payload: dict):
    """Fires a message to RabbitMQ in the background so the user doesn't wait."""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue="audit_trail", durable=True)
        channel.basic_publish(
            exchange="",
            routing_key="audit_trail",
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        connection.close()
    except Exception as e:
        print(f"Warning: Could not send to audit log. RabbitMQ might be down. Error: {e}")


@app.get("/health")
def health():
    """Liveness probe for Docker Compose and test_api.py."""
    return {"status": "ok", "service": "gateway"}


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    # ── Guard: reject non-image uploads before hitting any service
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Expected an image file (JPEG or PNG).")

    image_bytes = await file.read()
    files = {"file": (file.filename, image_bytes, file.content_type)}

    async with httpx.AsyncClient() as client:

        # ── STEP A: Classifier — get diagnosis + probabilities
        try:
            class_res = await client.post(CLASSIFIER_URL, files=files, timeout=30.0)
            class_res.raise_for_status()
            ml_data = class_res.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Classifier service is currently unavailable.")

        pneumonia_prob = ml_data.get("pneumonia_probability", 0.0)

        # ── STEP B: Severity — get risk level
        try:
            sev_res = await client.post(SEVERITY_URL, json={"pneumonia_probability": pneumonia_prob})
            sev_res.raise_for_status()
            triage_data = sev_res.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Severity service is currently unavailable.")

        # ── STEP C: Info — get compliance metadata (optional, don't block on failure)
        try:
            info_res = await client.get(INFO_URL)
            info_res.raise_for_status()
            meta_data = info_res.json()
        except Exception:
            meta_data = {}

    # ── STEP D: Assemble flat response (matches test_api.py schema exactly)
    final_response = {
        "diagnosis":             ml_data.get("diagnosis"),
        "confidence":            ml_data.get("confidence"),
        "pneumonia_probability": pneumonia_prob,
        "risk_level":            triage_data.get("risk_level"),
        "disclaimer":            DISCLAIMER,
        # Extended metadata (not tested but useful for frontend)
        "metadata":              meta_data,
    }

    # ── STEP E: Fire-and-forget audit log
    background_tasks.add_task(send_to_audit_log, final_response)

    return final_response