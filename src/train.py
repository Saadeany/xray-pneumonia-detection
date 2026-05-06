import torch
import torch.nn as nn
from sklearn.metrics import recall_score
import json
import os

from model import get_model
from dataset import get_dataloaders
from evaluate import evaluate_model

# --- CONFIGURATION ---
DATA_DIR = "data/chest_xray/chest_xray"
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "xray_classifier.pth"
METRICS_SAVE_PATH = "training_results.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Starting training pipeline on device: {device.upper()}")

    # 1. Initialize Architecture & Data
    model = get_model().to(device)
    train_loader, val_loader = get_dataloaders(DATA_DIR)

    # 2. Setup Optimizer & Loss (Removed double-weighting)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # 3. Tracking Metrics
    history = {"train_loss": [], "val_loss": [], "val_recall": []}
    best_recall = 0.0

    # 4. Training Loop
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_y_true, val_y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_recall = recall_score(val_y_true, val_y_pred, pos_label=1)

        history["val_loss"].append(avg_val_loss)
        history["val_recall"].append(val_recall)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Recall: {val_recall:.4f}")

        # --- MODEL CHECKPOINTING ---
        # Only save the model if it beat the previous best recall score
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" New best model saved! (Recall: {best_recall:.4f})")

    # 5. Final Evaluation on the best model
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    final_y_true, final_y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            final_y_true.extend(labels.numpy())
            final_y_pred.extend(preds.cpu().numpy())

    final_metrics = evaluate_model(final_y_true, final_y_pred)
    history["final_report"] = final_metrics

    # Save metrics to JSON
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nPipeline complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
