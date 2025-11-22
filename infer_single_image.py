# infer_disease_image.py

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from multi_class_cnn import (
    SimpleCNNMulti,
    DEVICE,
    IMG_SIZE,
    SAVE_BEST,
)

# Same normalization as validation
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

def load_model_and_classes():
    ckpt = torch.load(SAVE_BEST, map_location=DEVICE)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    model = SimpleCNNMulti(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, class_names

def predict_image(model, class_names, image_path: str, topk: int = 3):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().ravel()  # (C,)

    # Top-k predictions
    topk = min(topk, len(class_names))
    top_indices = probs.argsort()[::-1][:topk]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    return top_classes, top_probs

def main():
    parser = argparse.ArgumentParser(description="Infer disease type on a single leaf image")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print(f"Loading model from: {SAVE_BEST}")
    model, class_names = load_model_and_classes()

    top_classes, top_probs = predict_image(model, class_names, str(img_path), topk=3)

    print("\n========== DISEASE PREDICTION RESULT ==========")
    print(f"Image: {img_path}")
    print(f"Top-1: {top_classes[0]}  ({top_probs[0]*100:.2f}% confidence)")
    if len(top_classes) > 1:
        print("\nTop-k predictions:")
        for cls, p in zip(top_classes, top_probs):
            print(f"  - {cls:30s} : {p*100:.2f}%")
    print("===============================================")

if __name__ == "__main__":
    main()
