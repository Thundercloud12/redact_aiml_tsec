import streamlit as st
import hashlib
from PIL import Image
import io
import os
import numpy as np
import cv2
from skimage.measure import shannon_entropy

# ===============================================================
# BASIC SECURITY FUNCTIONS
# ===============================================================

def generate_sha256_hash(image_bytes):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_bytes)
    return sha256_hash.hexdigest()


# ===============================================================
# VISUAL CORRUPTION DETECTORS (NEW + IMPORTANT)
# ===============================================================

def detect_color_banding(path, threshold=40):
    try:
        img = cv2.imread(path)
        if img is None:
            return True
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)
        diffs = np.abs(np.diff(row_means))

        return np.max(diffs) > threshold
    except:
        return True


def detect_low_entropy(path, threshold=3.0):
    try:
        img = Image.open(path).convert("L")
        ent = shannon_entropy(np.array(img))
        return ent < threshold
    except:
        return True


def detect_unusual_color_distribution(path, threshold=5):
    try:
        img = cv2.imread(path)
        if img is None:
            return True

        b, g, r = cv2.split(img)

        for ch in [b, g, r]:
            if np.var(ch) < threshold:
                return True
        return False
    except:
        return True


def detect_extreme_blur(path, threshold=10):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        return var < threshold
    except:
        return True


def jpeg_decode_valid(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img is not None
    except:
        return False


# ===============================================================
# EXISTING FORMAT + CORRUPTION CHECKS
# ===============================================================

def validate_image_format(image_file):
    try:
        img = Image.open(image_file)
        img.verify()

        image_file.seek(0)
        img = Image.open(image_file)

        # force load
        img.load()

        # pixel test
        if img.mode not in ['RGB', 'RGBA', 'L']:
            img_test = img.convert("RGB")
        else:
            img_test = img

        pixels = list(img_test.getdata())

        if len(pixels) == 0:
            return False, "Pixel data empty", None

        for px in pixels[:100]:
            if isinstance(px, tuple):
                if any([c < 0 or c > 255 for c in px]):
                    return False, "Pixel out of range", None

        extrema = img.getextrema()

        format_info = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height
        }

        return True, format_info, img

    except Exception as e:
        return False, str(e), None


# ===============================================================
# MAIN VALIDATION WRAPPER INCLUDING VISUAL CORRUPTION CHECKS
# ===============================================================

def check_image_corruption(uploaded_file):

    # Save uploaded file temporarily for CV2-based checks
    temp_path = "temp_uploaded_image"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    uploaded_file.seek(0)  # Reset pointer

    results = {
        'hash': None,
        'format_valid': False,
        'not_empty': False,
        'not_corrupted': False,
        'format_info': {},
        'errors': [],
        'warnings': [],
        'visual_corruption': False
    }

    # Empty check
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size == 0:
        results['errors'].append("Image file is empty.")
        return results

    results['not_empty'] = True
    results['file_size'] = file_size

    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    results['hash'] = generate_sha256_hash(image_bytes)

    # Deep format validation
    is_valid, info, img = validate_image_format(uploaded_file)
    if not is_valid:
        results['errors'].append(f"Image validation failed: {info}")
    else:
        results['format_valid'] = True
        results['format_info'] = info
        results['image'] = img
        results['not_corrupted'] = True

    # ============================================================
    # 🔥 ADVANCED VISUAL CORRUPTION LAYER (what you're missing)
    # ============================================================

    banding = detect_color_banding(temp_path)
    entropy_low = detect_low_entropy(temp_path)
    blur = detect_extreme_blur(temp_path)
    color_distortion = detect_unusual_color_distribution(temp_path)
    jpg_decode = jpeg_decode_valid(temp_path)

    if banding:
        results['visual_corruption'] = True
        results['errors'].append("Severe color banding detected (image visually corrupted).")

    if entropy_low:
        results['visual_corruption'] = True
        results['errors'].append("Low entropy — image appears unnatural or glitchy.")

    if blur:
        results['warnings'].append("Image extremely blurred — may be visually corrupted.")

    if color_distortion:
        results['visual_corruption'] = True
        results['errors'].append("Unnatural color distribution detected.")

    if not jpg_decode:
        results['visual_corruption'] = True
        results['errors'].append("JPEG block decode check failed (truncated or glitched).")

    # If ANY visual corruption found → mark corrupted
    if results['visual_corruption']:
        results['not_corrupted'] = False

    return results


# ===============================================================
# STREAMLIT UI
# ===============================================================

def main():
    st.set_page_config(page_title="Image Integrity Checker", page_icon="🔒")

    st.title("🔒 Advanced Image Integrity Checker")
    st.write("Now includes **visual glitch detection**, color banding checks, entropy checks, and more.")

    uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg','bmp','gif','webp','tiff'])

    if uploaded_file:
        uploaded_file.seek(0)
        results = check_image_corruption(uploaded_file)

        st.subheader("🔍 Integrity Results")

        st.write("**SHA-256 Hash**:", results['hash'])

        # Basic checks
        if results['format_valid']:
            st.success("Format valid")
        else:
            st.error("Format invalid")

        if results['not_empty']:
            st.success("File not empty")
        else:
            st.error("File empty")

        if results['not_corrupted']:
            st.success("Not corrupted")
        else:
            st.error("Corrupted or visually corrupted")

        # Errors
        if results['errors']:
            st.subheader("❌ Errors")
            for err in results['errors']:
                st.error(err)

        # Warnings
        if results['warnings']:
            st.subheader("⚠️ Warnings")
            for w in results['warnings']:
                st.warning(w)

        # Image Preview
        if 'image' in results and results['not_corrupted']:
            st.image(results['image'], caption="Uploaded Image", use_container_width=True)
        else:
            st.write("Image cannot be displayed due to corruption.")
    else:
        st.info("Upload an image to begin integrity checks")


if __name__ == "__main__":
    main()
