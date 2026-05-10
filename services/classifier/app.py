import os
import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms, models

# Initialize FastAPI application
app = FastAPI(title="X-Ray Pneumonia Classifier API")

# Output classes for prediction
CLASSES = ["NORMAL", "PNEUMONIA"]

# ── Robust Path Resolution: Always finds the model no matter where Docker runs it
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray_classifier.pth")


def load_model(path: str):
    """Loads the ResNet18 model trained for binary classification."""

    # Create ResNet18 architecture
    model = models.resnet18(weights=None)
    # Replace final layer for binary classification
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Ensure model file exists before loading
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[STARTUP ERROR] Model not found at: {path}\n"
            f"Please ensure 'xray_classifier.pth' is inside the classifier folder."
        )

    # map_location="cpu" ensures it runs cleanly in our lightweight Docker container
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# Load the model exactly once when the server boots
model = load_model(MODEL_PATH)

# Image preprocessing pipeline (matches the training dataset)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # Adding standard ImageNet normalization to match ResNet's pre-training
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.get("/health")
def health():
    """Liveness probe for Docker Compose."""
    return {"status": "ok", "service": "classifier", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns the pneumonia probability."""
    # Guard: Reject non-image uploads instantly
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Expected an image file.")

    try:
        raw_bytes = await file.read()
        # Convert uploaded image into RGB format
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Cannot decode image. Send a valid JPEG/PNG.")

    # Apply transforms and add the batch dimension [1, 3, 224, 224]
    tensor = preprocess(img).unsqueeze(0)

    # Run inference without tracking gradients to save memory and speed up response
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pneumonia_prob = probs[1].item()
    max_confidence = probs.max().item()

    # Threshold 0.4: Prioritizing recall for medical screening
    pred = 1 if pneumonia_prob > 0.4 else 0

    return {
        "diagnosis": CLASSES[pred],
        "confidence": round(max_confidence, 4),
        "pneumonia_probability": round(pneumonia_prob, 4)
    }