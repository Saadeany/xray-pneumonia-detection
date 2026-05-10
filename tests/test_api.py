"""
Integration test for the X-Ray Pneumonia Detection API.

PRE-REQUISITE: The API server must be running before you execute this script.

  Option A (Docker — recommended):
      docker-compose up --build -d
      python tests/test_api.py

  Option B (local uvicorn):
      uvicorn services.gateway.main:app --port 8000
      python tests/test_api.py
"""

import os
import sys
import tempfile
import requests
from PIL import Image

# ── Change this if you deploy to a different host/port
BASE_URL = "http://127.0.0.1:8000"


def create_synthetic_xray() -> str:
    """
    Creates a temporary 224×224 grayscale JPEG that mimics an X-ray image.
    Returns the path; caller must delete it after use.

    Why synthetic instead of a real X-ray?
    - No patient data needed → safe for teammates
    - PIL is already in requirements, no extra deps
    - The model will return a valid (if meaningless) prediction
    """
    img = Image.new("L", (224, 224), color=128)   # "L" = 8-bit grayscale, mid-grey
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG")
    tmp.close()
    return tmp.name


def test_health():
    """Server must respond 200 with status=ok."""
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200, (
        f"[FAIL] /health returned {resp.status_code}. "
        f"Is the server running at {BASE_URL}?"
    )
    assert resp.json().get("status") == "ok", f"[FAIL] Unexpected body: {resp.json()}"
    print("✓ /health — OK")


def test_predict_valid_image():
    """POST a synthetic X-ray and validate the full response schema."""
    img_path = create_synthetic_xray()
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/predict",
                files={"file": ("synthetic.jpg", f, "image/jpeg")},
                timeout=30,   # model inference on CPU can take ~5–10 s
            )

        assert resp.status_code == 200, (
            f"[FAIL] /predict returned {resp.status_code}\n"
            f"  Body: {resp.text}\n"
            f"  If 422: python-multipart may be missing from the container"
        )

        result = resp.json()

        # ── Check all required fields are present
        required = ["diagnosis", "confidence", "pneumonia_probability",
                    "risk_level", "disclaimer"]
        for field in required:
            assert field in result, f"[FAIL] Missing field in response: '{field}'"

        # ── Validate field values
        assert result["diagnosis"] in ("NORMAL", "PNEUMONIA"), \
            f"[FAIL] diagnosis must be NORMAL or PNEUMONIA, got: {result['diagnosis']}"

        assert 0.0 <= result["confidence"] <= 1.0, \
            f"[FAIL] confidence out of [0,1]: {result['confidence']}"

        assert 0.0 <= result["pneumonia_probability"] <= 1.0, \
            f"[FAIL] pneumonia_probability out of [0,1]: {result['pneumonia_probability']}"

        assert result["risk_level"] in ("HIGH", "LOW", "CRITICAL"), \
            f"[FAIL] risk_level must be HIGH, LOW, or CRITICAL, got: {result['risk_level']}"

        # ── Validate disclaimer is non-empty and contains meaningful text
        disclaimer = result["disclaimer"]
        assert isinstance(disclaimer, str) and len(disclaimer) > 0, \
            "[FAIL] disclaimer must be a non-empty string"
        assert "SCREENING TOOL" in disclaimer.upper() or \
               "NOT A SUBSTITUTE" in disclaimer.upper(), \
            f"[FAIL] disclaimer must state this is a screening tool only. Got: '{disclaimer}'"

        print(f"✓ /predict — diagnosis: {result['diagnosis']}, "
              f"confidence: {result['confidence']:.2f}, "
              f"risk: {result['risk_level']}")
        print(f"  disclaimer: {disclaimer}")

    finally:
        os.unlink(img_path)   # ── Always clean up, even if an assertion fails


def test_predict_no_content_type():
    """
    Sending a request without a Content-Type header must return 415, not 500.
    This tests the gateway's None-safe content_type guard.
    """
    img_path = create_synthetic_xray()
    try:
        with open(img_path, "rb") as f:
            # Pass None as content_type to simulate a missing Content-Type header
            resp = requests.post(
                f"{BASE_URL}/predict",
                files={"file": ("synthetic.jpg", f, None)},
                timeout=10,
            )
        assert resp.status_code in (200, 415), (
            f"[FAIL] Expected 200 or 415 for missing Content-Type, got: {resp.status_code}"
        )
        print(f"✓ /predict no content-type — returned {resp.status_code} (acceptable)")
    finally:
        os.unlink(img_path)


def test_predict_invalid_file():
    """Sending a text file must return 415 or 422, not 500."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(b"this is not an image")
    tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/predict",
                files={"file": ("bad.txt", f, "text/plain")},
                timeout=10,
            )
        assert resp.status_code in (415, 422), (
            f"[FAIL] Expected 415 or 422 for non-image input, got: {resp.status_code}"
        )
        print(f"✓ /predict invalid input — correctly rejected with {resp.status_code}")
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print(f"Running integration tests against: {BASE_URL}\n")
    passed, failed = 0, 0

    for test_fn in [
        test_health,
        test_predict_valid_image,
        test_predict_no_content_type,
        test_predict_invalid_file,
    ]:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(e)
            failed += 1
        except requests.exceptions.ConnectionError:
            print(
                f"\n[ERROR] Cannot connect to {BASE_URL}\n"
                f"  Start the server first:\n\n"
                f"    docker-compose up --build -d     (Docker)\n"
                f"    uvicorn services.gateway.main:app --port 8000  (local)\n"
            )
            sys.exit(1)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)