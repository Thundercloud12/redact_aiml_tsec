# nodes/image_input.py
import os

def run(inputs: dict):
    """
    Expects inputs: { "image_path": "<absolute path or url>" }
    Returns { "image_path": "<abs path>" }
    """
    image_path = inputs.get("image_path")
    if not image_path:
        raise ValueError("image_path required")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return {"image_path": image_path}
