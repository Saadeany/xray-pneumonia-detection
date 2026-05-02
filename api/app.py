import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models

app = FastAPI(title="X-Ray Pneumonia Detection API")

CLASSES = ["NORMAL", "PNEUMONIA"]

# ── Path is resolved relative to THIS file, not the process working directory.
# ── Inside Docker: __file__ = /app/api/app.py → MODEL_PATH = /app/api/model.pth ✓
# ── Locally with `uvicorn api.app:app`: same logic applies ✓
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")


def load_model(path: str):
    # ── Architecture must exactly match src/train.py: resnet18 + 2-class head
    model = models.resnet18(weights=None)   # weights=None avoids deprecated pretrained=True warning
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(path):
        # ── Crash loudly at startup so teammates see a clear error immediately
        # ── instead of a confusing 500 on the first request
        raise FileNotFoundError(
            f"[STARTUP ERROR] model.pth not found at: {path}\n"
            f"  If running locally:  make sure api/model.pth exists\n"
            f"  If inside Docker:    rebuild the image (model is baked in via COPY)"
        )

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── Loaded once at startup — slow path happens here, not per-request
model = load_model(MODEL_PATH)

# ── X-rays are grayscale; Grayscale(3) converts 1-channel → 3-channel
# ── so the ResNet (which expects 3 channels) gets the right input shape
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


@app.get("/health")
def health():
    """Liveness check — used by docker-compose healthcheck and load balancers."""
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Guard: reject non-image uploads before reading the full file
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"[INPUT ERROR] Expected image/*, got: {file.content_type}"
        )

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        # ── PIL could not decode the bytes — file is corrupted or wrong format
        raise HTTPException(
            status_code=422,
            detail="[INPUT ERROR] Cannot decode image. Send a valid JPEG or PNG."
        )

    tensor = preprocess(img).unsqueeze(0)   # shape → [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pneumonia_prob = probs[1].item()
    # ── Threshold 0.4 (not 0.5) to favour recall: catching pneumonia matters
    # ── more than avoiding false positives for a screening tool
    pred = 1 if pneumonia_prob > 0.4 else 0

    return {
        "diagnosis":             CLASSES[pred],
        "confidence":            round(float(probs.max()), 4),
        "pneumonia_probability": round(pneumonia_prob, 4),
        "risk_level":            "HIGH" if pred == 1 else "LOW",
        "disclaimer":            "Screening tool only. Not a medical diagnosis.",
    }