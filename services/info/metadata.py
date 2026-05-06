from fastapi import FastAPI

app = FastAPI(title="Medical Info & Compliance Service")

@app.get("/health")
def health():
    """Liveness probe for Docker Compose."""
    return {"status": "ok", "service": "info"}

@app.get("/metadata")
def get_metadata():
    """
    Returns the static system information, model architecture details,
    and mandatory clinical disclaimers.
    """
    return {
        "system_version": "1.0.0",
        "model_architecture": "ResNet18 (ImageNet Pre-trained)",
        "training_dataset": "Kaggle Chest X-Ray Pneumonia Dataset",
        "clinical_disclaimer": (
            "WARNING: This AI system is an experimental screening tool only. "
            "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
            "All predictions must be reviewed by a certified radiologist."
        ),
        "compliance": "Audit logging enabled via RabbitMQ."
    }