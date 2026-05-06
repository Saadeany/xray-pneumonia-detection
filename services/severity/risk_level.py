from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Clinical Severity & Triage API")

# ── 1. Define the Expected Input
class PredictionInput(BaseModel):
    # Expect a float between 0.0 and 1.0
    pneumonia_probability: float = Field(..., ge=0.0, le=1.0)

# ── 2. Define the Output Structure
class TriageResponse(BaseModel):
    risk_level: str
    triage_category: str
    recommended_action: str

@app.get("/health")
def health():
    """Liveness probe for Docker Compose."""
    return {"status": "ok", "service": "severity"}

@app.post("/evaluate", response_model=TriageResponse)
def evaluate_severity(data: PredictionInput):
    """
    Takes the raw probability from the ML Classifier and maps it
    to actionable clinical triage levels.
    """
    prob = data.pneumonia_probability

    # ── Clinical Threshold Logic
    # (These can be easily updated without retraining the ML model)
    if prob < 0.40:
        return TriageResponse(
            risk_level="LOW",
            triage_category="Routine",
            recommended_action="Standard queue for radiologist review. No immediate flags."
        )
    elif prob < 0.70:
        return TriageResponse(
            risk_level="MODERATE",
            triage_category="Elevated",
            recommended_action="Flag for priority radiologist review within 4 hours."
        )
    elif prob < 0.90:
        return TriageResponse(
            risk_level="HIGH",
            triage_category="Urgent",
            recommended_action="Immediate radiologist review required. Potential acute infection."
        )
    else:
        return TriageResponse(
            risk_level="CRITICAL",
            triage_category="Emergency",
            recommended_action="URGENT: Alert attending physician. High confidence of severe pulmonary opacity."
        )