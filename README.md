# X-Ray Pneumonia Detection

A chest X-ray classifier that detects pneumonia using transfer learning (ResNet18) served via a FastAPI REST API.

---

## Project Structure

```
xray-pneumonia-detection/
├── api/            # FastAPI app + trained model (model.pth)
├── data/           # Dataset (downloaded via setup_data.py)
├── notebooks/      # EDA.ipynb — exploratory data analysis
├── src/            # Training pipeline
│   ├── dataset.py  # Data loaders (train / val / test)
│   ├── model.py    # ResNet18 transfer-learning model
│   ├── train.py    # Training + validation loop
│   └── evaluate.py # Metrics helpers
├── tests/          # Integration tests (require running server)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Download Dataset

Create a Kaggle API token (`~/.kaggle/kaggle.json`) then run:

```bash
python setup_data.py
```

This downloads the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset into `data/chest_xray/` with `train/`, `val/`, and `test/` subdirectories.

---

## Step 3 — Train the Model

```bash
cd src
python train.py
```

Saves `api/model.pth` and `api/training_results.json` on completion.  
Training prints per-epoch **train loss**, **validation loss**, and **recall**, then runs a final test-set evaluation.

---

## Step 4 — Run the API

```bash
uvicorn api.app:app --reload
```

Open the interactive docs at <http://localhost:8000/docs>.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/predict` | Upload a chest X-ray (JPEG/PNG), returns diagnosis |

### Example Response

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 0.8732,
  "pneumonia_probability": 0.8732,
  "risk_level": "HIGH",
  "disclaimer": "Screening tool only. Not a medical diagnosis."
}
```

---

## Step 5 — Docker

```bash
# Build and start (production)
docker-compose up --build

# Background mode
docker-compose up -d

# Stream logs
docker-compose logs -f api
```

The container exposes port **8000** and includes a `/health` health-check.

---

## Step 6 — Tests

Integration tests require a running server:

```bash
# Option A — Docker
docker-compose up -d
python tests/test_api.py

# Option B — local uvicorn
uvicorn api.app:app --port 8000
python tests/test_api.py
```

CI smoke-test (no model required):

```bash
docker run -d --name test-ci -p 8001:8000 -e MODEL_TEST_MODE=1 xray-pneumonia-api
sleep 15
curl http://localhost:8001/health
docker rm -f test-ci
```

---

## Notebooks

Open `notebooks/EDA.ipynb` for exploratory data analysis: class distribution, sample images, pixel intensity histograms, augmentation previews, and post-training confusion matrix / F1 charts.
