# 🩻 X-Ray Pneumonia Detection System

> **Distributed AI-Powered Clinical Screening Platform**
> Lightweight, containerized pneumonia detection using **ResNet18 + FastAPI + RabbitMQ + React**, engineered for **high recall**, **clinical safety**, and scalable deployment.

---

## 📌 Overview

This project is a production-oriented **AI-powered chest X-ray pneumonia detection platform** designed to minimize false negatives in high-risk clinical scenarios.

Unlike conventional prototypes, this system is architected as a **microservice mesh**, separating inference, triage, metadata injection, and compliance logging into independently scalable services.

### Core Design Priorities:

* **Clinical Safety First** → Recall-optimized model
* **False Negative Reduction** → Strict validation-recall checkpointing
* **Scalable Infrastructure** → RabbitMQ queue buffering
* **Containerized Deployment** → Docker Compose orchestration
* **Audit Compliance** → Immutable logging architecture
* **CPU-Compatible Inference** → Lightweight deployment without GPU dependency

---

# 🧠 Machine Learning Engine

The predictive engine was mathematically fine-tuned to prioritize **Recall** over generic accuracy metrics to align with real-world diagnostic risk.

| Metric                       | Score                                       | Detail                                                           |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| **Model Architecture**       | ResNet18                                    | Pre-trained ImageNet backbone with custom binary classifier head |
| **Validation Recall**        | **95%**                                     | Tested on unseen held-out radiograph validation set              |
| **Class Imbalance Handling** | Dynamic WeightedRandomSampler               | Implemented directly at DataLoader level                         |
| **Overfitting Prevention**   | Early Stopping + Recall-Based Checkpointing | Triggered on validation recall spikes                            |
| **Inference Speed**          | Lightweight                                 | CPU-friendly deployment                                          |

> **Clinical Note:** Missing a pneumonia-positive case is more dangerous than over-flagging a healthy case, which is why Recall was prioritized.

---

# 🧩 Distributed Microservice Ecosystem

## Architecture Philosophy:

Each service performs one dedicated responsibility, improving modularity, observability, and scalability.

| Service             | Technology | Responsibility                                         |
| ------------------- | ---------- | ------------------------------------------------------ |
| **Gateway**         | FastAPI    | Secure API entry point for uploads and orchestration   |
| **Broker**          | RabbitMQ   | Message queue to prevent RAM spikes and absorb load    |
| **Classifier**      | PyTorch    | Runs `model.pth` for binary pneumonia inference        |
| **Severity Engine** | Python     | Converts raw probabilities into clinical triage levels |
| **Info Injector**   | Python     | Adds disclaimers, metadata, and model version          |
| **Audit Logger**    | Python     | Immutable compliance + audit trail                     |

---

## 🔄 Request Lifecycle

```plaintext
User Uploads X-Ray
       ↓
 FastAPI Gateway
       ↓
   RabbitMQ Queue
       ↓
 PyTorch Classifier
       ↓
 Severity Translation
       ↓
 Metadata Injection
       ↓
 Audit Logging
       ↓
 Frontend Response
```

---

# 🚀 Getting Started

## 📋 Prerequisites

Before deployment, ensure the following are installed:

* **Docker Desktop** (v24.0+)
* **Node.js** (v18.0+)
* **Git**

---

# ⚙️ Local Deployment

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Saadeany/xray-pneumonia-detection.git
cd xray-pneumonia-detection
```

---

## 2️⃣ Boot the Distributed Backend

Launch the full **6-container microservice mesh**.
The `--build` flag ensures all localized dependencies are freshly compiled.

```bash
docker-compose up --build
```

> Wait for:
>
> * FastAPI Gateway to expose **port 8000**
> * RabbitMQ to complete startup successfully

---

## 3️⃣ Boot the Frontend Client

Open a **new terminal window** to avoid blocking backend service logs.

```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Application Access Points

| Service            | URL                    |
| ------------------ | ---------------------- |
| Frontend Client    | http://localhost:5173  |
| FastAPI Gateway    | http://localhost:8000  |
| RabbitMQ Dashboard | http://localhost:15672 |

> Once both backend and frontend are running, the platform is fully operational.

---

# 📂 Repository Structure

```plaintext
xray-pneumonia-detection/
│
├── frontend/                    # React + Vite client
│
├── notebooks/                   # EDA, training, experimentation
│
├── services/
│   ├── classifier/              # PyTorch model inference service
│   │   └── model.pth
│   │
│   ├── gateway/                 # FastAPI upload + routing
│   │
│   ├── severity/                # Clinical triage logic
│   │
│   ├── info/                    # Metadata & disclaimer injection
│   │
│   └── logger/                  # Immutable audit logging
│
└── docker-compose.yml           # Full orchestration blueprint
```

---

# 🛡 Clinical Safety Standards

## Why Recall Matters:

In healthcare AI:

### False Negative:

**Dangerous** → Missed pneumonia diagnosis

### False Positive:

**Manageable** → Extra physician review

### Therefore:

**High Recall > High Accuracy**

---

# 📈 Model Validation Strategy

### Training Controls:

* WeightedRandomSampler
* Binary Cross Entropy
* Validation Recall Tracking
* Early Stopping
* Best Model Checkpointing

---

# 🔐 Compliance & Auditability

Every prediction is logged with:

* Timestamp
* Model Version
* Prediction Score
* Severity Label
* Medical Disclaimer

This ensures:

* Reproducibility
* Compliance
* Version transparency
* Clinical accountability

---

# 🤝 Team Collaboration Guidelines

## Branching Rules:

```plaintext
feature/classifier-upgrade
feature/gateway-auth
feature/frontend-dashboard
feature/logger-enhancement
```

---

## Mandatory Restrictions:

* ❌ No direct push to `main`
* ❌ No unreviewed `docker-compose.yml` changes
* ❌ No schema modifications without team approval

---

# 🌍 Future Enhancements

* Multi-class pneumonia severity grading
* Explainable AI (Grad-CAM heatmaps)
* JWT Authentication
* PostgreSQL audit storage
* Kubernetes deployment
* CI/CD integration
* Radiologist dashboard analytics

---

# ⭐ Final Note

This project is not just a model.
It is a **clinical-safe distributed inference ecosystem** built with production logic in mind.

If you’re building healthcare AI, architecture matters just as much as accuracy.
