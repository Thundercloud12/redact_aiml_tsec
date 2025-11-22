import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from evaluate import SimpleCNN, build_loaders, DEVICE, IMG_SIZE

MODEL_PATH = "best_model.pth"

def load_model():
    model = SimpleCNN().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def evaluate():
    # Build val dataloader only
    _, val_loader, _ = build_loaders("data/train", "data/valid", debug=False)

    model = load_model()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================= MODEL EVALUATION =================")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("====================================================")

if __name__ == "__main__":
    evaluate()
