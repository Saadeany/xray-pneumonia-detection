import torch
import torch.nn as nn
from sklearn.metrics import recall_score
import json
import os

from model import get_model
from dataset import get_dataloaders
from evaluate import evaluate_model

# ── CONFIGURATION
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data", "chest_xray")
EPOCHS          = 10
LEARNING_RATE   = 1e-3
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "src", "xray_classifier.pth")
METRICS_SAVE_PATH = os.path.join(BASE_DIR, "src", "training_results.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training pipeline on device: {device.upper()}")

    # ── 1. Data & Model
    # Fix: now returns 3 loaders — train, val, test
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)
    model = get_model().to(device)

    # ── 2. Optimizer — Fix: was only training model.fc, now also trains layer4 (which was unfrozen)
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4},   # fine-tune unfrozen block
        {"params": model.fc.parameters(),     "lr": 1e-3},   # train new classifier head
    ])

    # ── 3. Loss
    criterion = nn.CrossEntropyLoss()

    # ── 4. LR Scheduler — Fix: was completely missing, required by spec
    # Reduces LR when val recall stops improving (patience=2 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",        # maximizing recall
        patience=2,
        factor=0.5,
    )

    # ── 5. Metric Tracking
    history = {
        "train_loss": [],
        "val_loss":   [],
        "val_recall": [],
    }
    best_recall = 0.0

    # ── 6. Training Loop
    for epoch in range(EPOCHS):

        # ── TRAIN PHASE
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ── VALIDATION PHASE
        model.eval()
        val_loss   = 0.0
        val_y_true, val_y_pred, val_y_prob = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss += loss.item()

                # Fix: collect probabilities (needed for ROC-AUC + threshold tuning)
                probs           = torch.softmax(outputs, dim=1)
                pneumonia_probs = probs[:, 1].cpu().numpy()
                preds           = (pneumonia_probs > 0.40).astype(int)   # threshold 0.4

                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds)
                val_y_prob.extend(pneumonia_probs)

        avg_val_loss = val_loss / len(val_loader)
        val_recall   = recall_score(val_y_true, val_y_pred, pos_label=1, zero_division=0)

        history["val_loss"].append(avg_val_loss)
        history["val_recall"].append(val_recall)

        print(
            f"Epoch {epoch+1:>2}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Recall: {val_recall:.4f}"
        )

        # ── LR Scheduler step (based on val recall)
        scheduler.step(val_recall)

        # ── Save best model (recall-based checkpointing)
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  → New best model saved! (Recall: {best_recall:.4f})")

    # ── 7. Final Evaluation on TEST SET (Fix: was using val_loader)
    print("\nLoading best model for final evaluation on TEST SET...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_y_true, test_y_pred, test_y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            probs           = torch.softmax(outputs, dim=1)
            pneumonia_probs = probs[:, 1].cpu().numpy()
            preds           = (pneumonia_probs > 0.40).astype(int)

            test_y_true.extend(labels.numpy())
            test_y_pred.extend(preds)
            test_y_prob.extend(pneumonia_probs)

    # Fix: pass y_prob so evaluate_model can compute ROC-AUC + threshold tuning
    final_metrics = evaluate_model(test_y_true, test_y_pred, y_prob=test_y_prob)
    history["final_report"] = final_metrics

    # ── 8. Save full training history + metrics
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nPipeline complete.")
    print(f"  Model   → {MODEL_SAVE_PATH}")
    print(f"  Metrics → {METRICS_SAVE_PATH}")
    print(f"  Best Val Recall: {best_recall:.4f}")


if __name__ == "__main__":
    main()