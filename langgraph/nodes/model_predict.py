# nodes/model_predict.py
import tensorflow as tf
import numpy as np
import os

# Path to your finetuned Keras model file (update to real path)
MODEL_PATH = "/kaggle/working/mobilenet_v3_final.h5"  # <-- put the model file here or S3 URL

# lazy load
_MODEL = None
def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _MODEL

def run(inputs: dict):
    model = _load_model()
    image_tensor = inputs["image_tensor"]  # shape (1,H,W,C)
    preds = model.predict(image_tensor)
    # if final layer was sigmoid/1D -> adapt; here we assume 2-class softmax
    if preds.ndim == 2 and preds.shape[1] == 2:
        prob_healthy = float(preds[0,1])
        prob_diseased = float(preds[0,0])
        pred_label = 1 if prob_healthy > prob_diseased else 0
    else:
        # fallback: single prob means healthy probability
        prob_healthy = float(preds.ravel()[0])
        pred_label = 1 if prob_healthy >= 0.5 else 0
        prob_diseased = 1.0 - prob_healthy
    classes = {1: "Healthy", 0: "Diseased"}
    return {
        "pred_label": int(pred_label),
        "pred_class": classes[int(pred_label)],
        "prob_healthy": prob_healthy,
        "prob_diseased": prob_diseased,
        "preds_raw": preds.tolist()
    }
