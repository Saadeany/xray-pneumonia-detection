import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from model import get_model
from dataset import get_dataloaders, get_test_loader
from evaluate import evaluate
import json


train_losses = []
val_losses = []
recalls = []


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = get_model().to(device)

weights = torch.tensor([1.0, 2.0])  # NORMAL, PNEUMONIA
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

train_loader, val_loader = get_dataloaders("../data/chest_xray")

for epoch in range(5):
    # ── Training
    model.train()

    epoch_loss = 0
    num_batches = 0

    y_true, y_pred = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    recall = recall_score(y_true, y_pred, pos_label=1)
    recalls.append(recall)

    # ── Validation
    model.eval()
    val_loss  = 0
    val_batches = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss  += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Recall: {recall:.4f}")


results = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "recalls": recalls,
    "best_epoch": int(max(range(len(recalls)), key=lambda i: recalls[i])),
    "final_recall": float(recalls[-1])
}

with open("../api/training_results.json", "w") as f:
    json.dump(results, f)

torch.save(model.state_dict(), "../api/model.pth")
print("Model saved!")

# ── Final evaluation on the held-out test set
print("\n── Test set evaluation ──")
test_loader = get_test_loader("../data/chest_xray")

model.eval()
y_true_test, y_pred_test = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds.cpu().numpy())

evaluate(y_true_test, y_pred_test)

