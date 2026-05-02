import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from model import get_model
from dataset import get_dataloaders
import json


train_losses = []
recalls = []


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = get_model().to(device)

weights = torch.tensor([1.0, 2.0])  # NORMAL, PNEUMONIA
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

train_loader = get_dataloaders("../data/chest_xray")

for epoch in range(5):
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

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    recall = recall_score(y_true, y_pred, pos_label=1)
    recalls.append(recall)

    print(f"Epoch {epoch} - Loss: {avg_loss:.4f} - Recall: {recall:.4f}")


results = {
    "train_losses": train_losses,
    "recalls": recalls,
    "best_epoch": int(max(range(len(recalls)), key=lambda i: recalls[i])),
    "final_recall": float(recalls[-1])
}

with open("../api/training_results.json", "w") as f:
    json.dump(results, f)

torch.save(model.state_dict(), "../api/model.pth")
print("Model saved!")

