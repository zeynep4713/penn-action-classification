import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import accuracy_score, classification_report
from dataset import PoseDatasetFromNpy
from model import PoseLSTM
from collate_fn import collate_fn

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_and_evaluate(npy_folder, labels_json, batch_size=16, epochs=125, patience=10, min_delta=0.001):
    train_dataset = PoseDatasetFromNpy(npy_folder, labels_json, split="train")
    val_dataset = PoseDatasetFromNpy(npy_folder, labels_json, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = PoseLSTM(input_size=51, num_classes=len(train_dataset.encoder.classes_)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Replace StepLR with ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # because we monitor val_loss
        factor=0.5,        # reduce LR by half
        patience=4,        # wait 4 epochs before reducing LR
        min_lr=1e-6
        # verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0

        for x, lengths, y in train_loader:
            x, lengths, y = x.cuda(), lengths.cuda(), y.cuda()

            out = model(x, lengths)
            loss = criterion(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.cuda(), lengths.cuda(), y.cuda()

                out = model(x, lengths)
                loss = criterion(out, y)
                val_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # IMPORTANT: pass val_loss to scheduler
        scheduler.step(avg_val_loss)

        # ---------------- EARLY STOPPING ----------------
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "best_model.pth")

        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # ---------------- LOAD BEST MODEL ----------------
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # ---------------- FINAL EVALUATION ----------------
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, lengths, y in val_loader:
            x, lengths, y = x.cuda(), lengths.cuda(), y.cuda()
            out = model(x, lengths)

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nValidation Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.encoder.classes_))

    cm = confusion_matrix(all_labels, all_preds)
    class_names = train_dataset.encoder.classes_

    plt.figure(figsize=(11, 7.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate("Penn_Action/keypoints_yl", "Penn_Action/labels.json")