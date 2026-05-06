import os
import json
import pika
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Medical AI API Gateway")

# ── 1. CORS Configuration (Allows React Frontend to talk to this Gateway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Microservice Internal URLs (Docker Compose DNS mapping)
CLASSIFIER_URL = "http://classifier:8000/predict"
SEVERITY_URL = "http://severity:8000/evaluate"
INFO_URL = "http://info:8000/metadata"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")


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
            properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
        )
        connection.close()
    except Exception as e:
        print(f"Warning: Could not send to audit log. RabbitMQ might be down. Error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/analyze")
async def analyze_xray(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    The Master Endpoint. Orchestrates the flow between Classifier, Severity, and Info.
    """
    # 1. Read the image
    image_bytes = await file.read()
    files = {"file": (file.filename, image_bytes, file.content_type)}

    async with httpx.AsyncClient() as client:
        # ── STEP A: Ask the Classifier ──
        try:
            class_res = await client.post(CLASSIFIER_URL, files=files, timeout=10.0)
            class_res.raise_for_status()
            ml_data = class_res.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Classifier Service is currently unavailable.")

        pneumonia_prob = ml_data.get("pneumonia_probability", 0.0)

        # ── STEP B: Ask the Severity Engine ──
        try:
            sev_res = await client.post(SEVERITY_URL, json={"pneumonia_probability": pneumonia_prob})
            sev_res.raise_for_status()
            triage_data = sev_res.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Severity Service is currently unavailable.")

        # ── STEP C: Ask the Info/Compliance Engine ──
        try:
            info_res = await client.get(INFO_URL)
            info_res.raise_for_status()
            meta_data = info_res.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Info Service is currently unavailable.")

    # ── STEP D: Assemble the Final Master JSON ──
    final_response = {
        "status": "success",
        "results": {
            "diagnosis": ml_data.get("diagnosis"),
            "confidence": ml_data.get("confidence"),
            "pneumonia_probability": pneumonia_prob
        },
        "triage": triage_data,
        "metadata": meta_data
    }

    # ── STEP E: Send to Logger asynchronously ──
    background_tasks.add_task(send_to_audit_log, final_response)

    # ── STEP F: Return to React Frontend ──
    return final_response