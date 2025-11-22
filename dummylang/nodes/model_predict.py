# nodes/model_predict.py (FAKE MODEL FOR TESTING)

import random

def run(inputs: dict):
    # Simulate a model output (random)
    pred_label = random.choice([0, 1])
    classes = {1: "Healthy", 0: "Diseased"}

    prob_healthy = random.uniform(0.5, 1.0) if pred_label == 1 else random.uniform(0.0, 0.5)
    prob_diseased = 1 - prob_healthy

    return {
        "pred_label": pred_label,
        "pred_class": classes[pred_label],
        "prob_healthy": float(prob_healthy),
        "prob_diseased": float(prob_diseased),
        "preds_raw": [prob_diseased, prob_healthy]
    }
