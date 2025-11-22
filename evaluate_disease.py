# analyze_disease_model.py
"""
Comprehensive analyzer for the disease multi-class model.
Place this file next to train_disease_torch_fixed_amp.py and run:
    python analyze_disease_model.py
It will:
 - load best model (SAVE_BEST from train_disease_torch_fixed_amp.py)
 - evaluate on validation set
 - compute metrics, confusion matrix, per-class stats
 - list top confusion pairs and save example images into analysis_results/
 - save CSV reports under analysis_results/
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from PIL import Image

# Import objects from your training script
# Make sure train_disease_torch_fixed_amp.py is in same folder
from multi_class_cnn import (
    build_loaders,
    SAVE_BEST,
    DEVICE,
    TRAIN_DIR,
    VAL_DIR,
    BATCH_SIZE,
    DEBUG,
    IMG_SIZE,
    SimpleCNNMulti,
)

OUTDIR = Path("analysis_results")
TOP_CONF_PAIRS = 20        # how many confusion pairs to report/save
EXAMPLES_PER_PAIR = 8      # how many example images to save per top pair
CONFIDENCE_THRESHOLD = 0.5 # used for some diagnostics (not a decision threshold)

def load_checkpoint_and_model():
    ckpt = torch.load(SAVE_BEST, map_location=DEVICE)
    class_names = ckpt.get("class_names") or ckpt.get("class_to_idx").keys()
    class_names = list(class_names)
    num_classes = len(class_names)
    model = SimpleCNNMulti(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names

def run_inference(model, val_loader):
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels, paths in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C)
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

    if not all_probs:
        raise RuntimeError("No validation data found in loader")
    probs_all = np.vstack(all_probs)
    preds_all = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    return labels_all, preds_all, probs_all, all_paths

def per_class_confidence_analysis(labels_all, preds_all, probs_all, class_names):
    n_classes = len(class_names)
    stats = []
    for i in range(n_classes):
        idxs = np.where(labels_all == i)[0]
        if len(idxs) == 0:
            stats.append({
                "class_idx": i,
                "class_name": class_names[i],
                "support": 0,
                "acc": None,
                "avg_conf_correct": None,
                "avg_conf_incorrect": None,
                "frac_correct": None
            })
            continue
        preds_for_class = preds_all[idxs]
        probs_for_class = probs_all[idxs, :]
        correct_mask = preds_for_class == i
        frac_correct = float(correct_mask.sum()) / len(idxs)
        avg_conf_correct = float(probs_for_class[correct_mask, i].mean()) if correct_mask.sum() > 0 else 0.0
        avg_conf_incorrect = float(probs_for_class[~correct_mask, preds_for_class[~correct_mask]].mean()) if (~correct_mask).sum() > 0 else 0.0
        stats.append({
            "class_idx": i,
            "class_name": class_names[i],
            "support": int(len(idxs)),
            "acc": float(frac_correct),
            "avg_conf_correct": float(avg_conf_correct),
            "avg_conf_incorrect": float(avg_conf_incorrect),
            "frac_correct": float(frac_correct)
        })
    return pd.DataFrame(stats)

def top_confusion_pairs(cm, class_names, topk=20):
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt > 0:
                pairs.append((cnt, class_names[i], class_names[j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:topk]

def save_examples_for_pairs(pairs, true_indices, pred_indices, paths, outdir: Path, max_examples=8):
    # Map (true_idx,pred_idx) -> list of example paths
    pair_examples = defaultdict(list)
    for t,p,pt in zip(true_indices, pred_indices, paths):
        key = (int(t), int(p))
        if key in [(x[3], x[4]) for x in pairs]:
            pair_examples[key].append(pt)

    # create folders and copy images
    for cnt,true_name,pred_name,true_idx,pred_idx in pairs:
        key = (true_idx, pred_idx)
        examples = pair_examples.get(key, [])[:max_examples]
        target_dir = outdir / f"{true_idx:02d}_{sanitize(class_names[true_idx])}__to__{pred_idx:02d}_{sanitize(class_names[pred_idx])}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for i,p in enumerate(examples):
            try:
                img = Image.open(p).convert("RGB")
                # save thumbnail sized for quick view
                img.thumbnail((512,512))
                save_path = target_dir / f"ex_{i+1}_{Path(p).name}"
                img.save(save_path)
            except Exception as e:
                # fallback: copy
                try:
                    shutil.copy(p, target_dir / Path(p).name)
                except Exception:
                    pass

def sanitize(s: str):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120]

if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)

    print("Loading model and class names from:", SAVE_BEST)
    model, class_names = load_checkpoint_and_model()
    print("Found classes:", class_names)

    print("Building val loader...")
    _, val_loader, meta = build_loaders(TRAIN_DIR, VAL_DIR, batch_size=BATCH_SIZE, debug=DEBUG)
    print(f"Validation samples: {meta['n_val']}")

    print("Running inference on validation set...")
    y_true, y_pred, probs_all, paths = run_inference(model, val_loader)
    print("Done inference. Aggregating metrics...")

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== OVERALL METRICS ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro)   : {rec_macro:.4f}")
    print(f"F1-Score (macro) : {f1_macro:.4f}")

    # Per-class report
    print("\nGenerating per-class classification report...")
    clf_report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(clf_report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    print("Confusion matrix (raw) shape:", cm.shape)

    # Save raw predictions with metadata
    print("Saving predictions CSV...")
    pred_meta_df = pd.DataFrame({
        "path": paths,
        "true_idx": y_true,
        "true_label": [class_names[int(i)] for i in y_true],
        "pred_idx": y_pred,
        "pred_label": [class_names[int(i)] for i in y_pred],
        "pred_conf_max": probs_all.max(axis=1),
    })
    pred_meta_df.to_csv(OUTDIR / "predictions_with_meta.csv", index=False)

    # Top confusion pairs
    pairs = top_confusion_pairs(cm, class_names, topk=TOP_CONF_PAIRS)
    print("\nTop confusion pairs (count, true -> pred):")
    for cnt, true_name, pred_name, t_idx, p_idx in pairs:
        print(f"{cnt:4d} : {true_name}  ->  {pred_name}")

    # Save example images for top pairs
    print(f"\nSaving up to {EXAMPLES_PER_PAIR} example images per top pair into {OUTDIR}/examples/")
    save_examples_for_pairs(pairs, y_true, y_pred, paths, OUTDIR / "examples", max_examples=EXAMPLES_PER_PAIR)

    # Per-class confidence analysis
    print("\nPer-class confidence analysis...")
    conf_df = per_class_confidence_analysis(y_true, y_pred, probs_all, class_names)
    conf_df.to_csv(OUTDIR / "per_class_confidence.csv", index=False)
    print(conf_df.sort_values("support", ascending=False).head(20).to_string(index=False))

    # Summary CSV
    summary = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "n_val": int(len(y_true)),
        "n_classes": len(class_names)
    }
    pd.DataFrame([summary]).to_csv(OUTDIR / "summary.csv", index=False)

    # Quick diagnostics / suggestions
    print("\n=== DIAGNOSTICS & SUGGESTIONS ===")
    # 1) Check largest confusion pair sizes
    if pairs and pairs[0][0] > 50:
        print(f"- Large confusions found: top pair {pairs[0][1]} -> {pairs[0][2]} (count={pairs[0][0]}). Inspect saved examples in {OUTDIR / 'examples'}.")
    else:
        print("- No extremely large confusion pairs (top pair <= 50).")

    # 2) Check per-class supports for very small classes
    small_classes = conf_df[conf_df["support"] < 50]
    if not small_classes.empty:
        print(f"- Found {len(small_classes)} small classes (support < 50). Consider collecting more data or augmenting them.")
        print(small_classes[["class_idx", "class_name", "support"]].to_string(index=False))
    else:
        print("- No very small classes detected (support >= 50 for all).")

    # 3) Classes with low avg conf on correct examples
    low_conf_ok = conf_df[conf_df["avg_conf_correct"] < 0.6]
    if not low_conf_ok.empty:
        print("- Classes with low avg confidence even when correct (may be hard/ambiguous):")
        print(low_conf_ok[["class_idx", "class_name", "support", "avg_conf_correct"]].to_string(index=False))
    else:
        print("- Per-class avg confidence for correct predictions looks healthy (>=0.6).")

    # 4) Overall score check
    if f1_macro < 0.90:
        print("- Macro F1 < 0.90: consider stronger augmentation, unfreezing more backbone layers, or focal loss.")
    elif f1_macro < 0.95:
        print("- Macro F1 between 0.90 and 0.95: good. Consider targeted fixes for top confusion pairs described above.")
    else:
        print("- Macro F1 >= 0.95: excellent. Focus on label cleaning and edge-case examples if needed.")

    print("\nAll results saved to:", OUTDIR.resolve())
    print("Open analysis_results/predictions_with_meta.csv and analysis_results/examples/ to inspect images.")
