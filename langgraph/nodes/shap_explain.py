# nodes/shap_explain.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
from PIL import Image

# where to store outputs
OUT_DIR = "/kaggle/working/agri_explanations"
os.makedirs(OUT_DIR, exist_ok=True)

# load model lazily (same path used earlier)
MODEL_PATH = "/kaggle/working/mobilenet_v3_final.h5"
_MODEL = None
def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _MODEL

def run(inputs: dict):
    model = _load_model()
    image_tensor = inputs["image_tensor"]  # (1,H,W,C) float32 preprocessed
    # choose a small background for SHAP (use a few training images ideally; here we use the image itself repeated)
    # For speed, use GradientExplainer for TF models
    try:
        explainer = shap.GradientExplainer(model, image_tensor)  # fast approximate
        shap_values = explainer.shap_values(image_tensor)  # returns list per output class
    except Exception as e:
        # fallback to GradientTape saliency (gradients wrt class logit)
        img = tf.cast(image_tensor, tf.float32)
        pred_index = tf.argmax(model(img), axis=1)[0].numpy()
        with tf.GradientTape() as tape:
            tape.watch(img)
            logits = model(img)
            score = logits[:, pred_index]
        grads = tape.gradient(score, img)[0].numpy()
        # normalize grads to 0..1
        sal = np.abs(grads).mean(axis=2)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        # save heatmap
        heatmap = (sal * 255).astype('uint8')
        heatpath = os.path.join(OUT_DIR, "saliency_heatmap.png")
        Image.fromarray(heatmap).resize((224,224)).save(heatpath)
        return {"shap_path": heatpath, "method": "gradients", "note": str(e)}

    # shap_values shape: [n_classes][1,H,W,C] or similar. We'll aggregate channels
    # We'll compute heatmap for the predicted class (assuming pred_label provided)
    pred_label = inputs.get("pred_label", None)
    if pred_label is None:
        pred_label = int(np.argmax(model.predict(image_tensor)[0]))
    # shap_values[pred_label] -> shape (1,H,W,C)
    sv = shap_values[pred_label][0] if isinstance(shap_values, list) else shap_values[0][0]
    heat = np.mean(np.abs(sv), axis=2)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_rgb = (np.uint8(255 * heat))
    heatpath = os.path.join(OUT_DIR, f"shap_heatmap_{os.path.basename(inputs.get('image_path','img'))}.png")
    # save heatmap with matplotlib for overlay
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(heat, cmap='jet')
    plt.savefig(heatpath, bbox_inches='tight', pad_inches=0)
    plt.close()
    return {"shap_path": heatpath, "method": "shap_gradient"}
