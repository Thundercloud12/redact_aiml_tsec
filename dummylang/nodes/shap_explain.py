# nodes/shap_explain.py (FAKE SHAP FOR TESTING)
import os
import numpy as np
from PIL import Image

OUT_DIR = "/kaggle/working/agri_explanations"
os.makedirs(OUT_DIR, exist_ok=True)

def run(inputs: dict):
    # Create a dummy heatmap (red square)
    heatmap = np.zeros((224,224,3), dtype=np.uint8)
    heatmap[:,:,0] = 255   # red

    heat_path = os.path.join(
        OUT_DIR,
        "dummy_heatmap_" + os.path.basename(inputs["image_path"]) + ".png"
    )
    Image.fromarray(heatmap).save(heat_path)

    return {"shap_path": heat_path, "method": "dummy_test_mode"}
