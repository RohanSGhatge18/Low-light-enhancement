# streamlit_app.py
"""
Improved Low-Light Enhancer Streamlit app
- Removes Streamlit deprecation warnings (use_container_width)
- Better denoising + sharpening
- Auto-adjust parameters based on image brightness/noise
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter
import io

# -----------------------
# Helpers
# -----------------------
def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_bgr: np.ndarray):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def image_to_bytes(img_pil: Image.Image, fmt="JPEG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt, quality=95)
    return buf.getvalue()

# Unsharp mask (PIL) alternative for fine detail
def unsharp_pil(pil_img: Image.Image, amount=1.0, radius=1, threshold=0):
    # PIL's built-in unsharp mask
    return pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(150*amount), threshold=threshold))

# -----------------------
# Enhancement building blocks
# -----------------------
def denoise_nlmeans(img_bgr, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21, bilateral=False):
    """Denoise using OpenCV colored NLMeans. Optionally apply bilateral filter for edge preservation."""
    out = img_bgr.copy()
    if bilateral:
        # bilateral first can help preserve edges (optional)
        out = cv2.bilateralFilter(out, d=5, sigmaColor=75, sigmaSpace=75)
    try:
        out = cv2.fastNlMeansDenoisingColored(out, None, float(h), float(hColor), templateWindowSize, searchWindowSize)
    except Exception:
        out = img_bgr.copy()
    return out

def apply_clahe_lab(img_bgr, clip=2.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def gamma_correction(img_bgr, gamma=1.2):
    g = float(gamma)
    if g <= 0:
        g = 1.0
    invGamma = 1.0 / g
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)

def sharpen_cv2(img_bgr, strength=1.0):
    # unsharp-like kernel that scales with strength
    try:
        base = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        if strength != 1.0:
            identity = np.zeros_like(base); identity[1,1] = 1.0
            s = float(max(0.0, min(strength, 5.0)))
            kernel = identity * (1.0 - (s - 1.0) * 0.25) + base * (s - 1.0) * 0.25
        else:
            kernel = base
        return cv2.filter2D(img_bgr, -1, kernel)
    except Exception:
        return img_bgr

# Combined full-enhance pipeline
def full_enhance(img_bgr, denoise_on=True, h=10, hColor=10, clip=2.0, tile=8, gamma=1.2, sharp=1.0, bilateral=False):
    out = img_bgr.copy()
    if denoise_on:
        # use a slightly larger search window for stronger, quieter result
        sw = 31 if h >= 15 else 21
        out = denoise_nlmeans(out, h=h, hColor=hColor, templateWindowSize=7, searchWindowSize=sw, bilateral=bilateral)
    out = apply_clahe_lab(out, clip=clip, tile=tile)
    out = gamma_correction(out, gamma=gamma)
    out = sharpen_cv2(out, strength=sharp)
    # final light unsharp in PIL for nicer detail control (convert temporarily)
    try:
        pil = cv2_to_pil(out)
        pil = unsharp_pil(pil, amount=max(0.0, min(sharp, 2.0)), radius=1, threshold=0)
        out = pil_to_cv2(pil)
    except Exception:
        pass
    return out

# -----------------------
# Auto-Adjustment rules
# -----------------------
def auto_suggest_params(pil_img):
    """Estimate brightness and noise and return suggested params."""
    small = pil_img.resize((200,200))
    arr = np.array(small).astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    avg = float(gray.mean())
    std = float(gray.std())

    # defaults
    params = {
        "denoise_on": True,
        "h": 10.0,
        "hColor": 10.0,
        "clip": 2.0,
        "tile": 8,
        "gamma": 1.2,
        "sharp": 1.0,
        "bilateral": False
    }

    # brightness heuristics
    if avg < 35:
        params["gamma"] = 1.6
        params["clip"] = 3.5
        params["h"] = max(params["h"], 18.0)
        params["hColor"] = max(params["hColor"], 18.0)
        params["sharp"] = 1.1
    elif avg < 70:
        params["gamma"] = 1.35
        params["clip"] = 2.5
        params["h"] = max(params["h"], 12.0)
        params["hColor"] = max(params["hColor"], 12.0)
    else:
        # already bright image: use softer corrections
        params["gamma"] = 1.05
        params["clip"] = 1.8
        params["h"] = 8.0
        params["hColor"] = 8.0

    # noise heuristics (stddev)
    if std > 40:
        params["h"] = max(params["h"], 20.0)
        params["hColor"] = max(params["hColor"], 20.0)
        params["bilateral"] = True
    elif std > 25:
        params["h"] = max(params["h"], 14.0)
        params["hColor"] = max(params["hColor"], 14.0)

    return params

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Low Light Enhancer", layout="wide")
st.title("Performance in Dark and Low-Light Environment")
st.write("Original | Denoised-only | Fully Enhanced — tweak sliders or enable Auto-adjust for one-click suggestions.")

st.markdown("---")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_cv2(pil_img)

# Sidebar controls
st.sidebar.header("Settings")
auto_adj = st.sidebar.checkbox("Auto-adjust parameters", value=True)

# If auto-adjust is enabled, compute suggestions and pre-fill sliders
if auto_adj:
    suggested = auto_suggest_params(pil_img)
else:
    suggested = None

denoise_on = st.sidebar.checkbox("Apply Denoise (NLMeans)", value=(suggested["denoise_on"] if suggested else True))
h = st.sidebar.slider("Denoise h", 0.0, 30.0, float(suggested["h"] if suggested else 10.0))
hColor = st.sidebar.slider("Denoise hColor", 0.0, 30.0, float(suggested["hColor"] if suggested else 10.0))
clip = st.sidebar.slider("CLAHE clip limit", 1.0, 5.0, float(suggested["clip"] if suggested else 2.0))
tile = st.sidebar.slider("CLAHE tile grid", 2, 16, int(suggested["tile"] if suggested else 8))
gamma = st.sidebar.slider("Gamma", 0.5, 3.0, float(suggested["gamma"] if suggested else 1.2))
sharp = st.sidebar.slider("Sharpen strength", 0.0, 3.0, float(suggested["sharp"] if suggested else 1.0))
bilateral = st.sidebar.checkbox("Use bilateral filter for stronger edge preservation", value=(suggested["bilateral"] if suggested else False))

st.sidebar.markdown("---")
st.sidebar.write("Tip: enable Auto-adjust for quick results on very dark/noisy images.")

# Compute images
# Denoised-only (always compute for comparison)
denoised_bgr = denoise_nlmeans(img_bgr, h=h, hColor=hColor, templateWindowSize=7, searchWindowSize=31, bilateral=bilateral)
denoised_pil = cv2_to_pil(denoised_bgr)

# Enhanced (apply denoise depending on toggle)
enhanced_bgr = full_enhance(img_bgr, denoise_on=denoise_on, h=h, hColor=hColor, clip=clip, tile=tile, gamma=gamma, sharp=sharp, bilateral=bilateral)
enhanced_pil = cv2_to_pil(enhanced_bgr)

orig_pil = pil_img

# Layout display (three columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original")
    st.image(orig_pil, use_container_width=True)           # fixed: use_container_width
    st.download_button("Download Original", data=image_to_bytes(orig_pil), file_name="original.jpg", mime="image/jpeg")

with col2:
    st.subheader("Denoised (NLMeans)")
    st.image(denoised_pil, use_container_width=True)
    st.download_button("Download Denoised", data=image_to_bytes(denoised_pil), file_name="denoised.jpg", mime="image/jpeg")

with col3:
    st.subheader("Fully Enhanced")
    st.image(enhanced_pil, use_container_width=True)
    st.download_button("Download Enhanced", data=image_to_bytes(enhanced_pil), file_name="enhanced.jpg", mime="image/jpeg")

st.markdown("---")

# Side-by-side combined comparison and download
st.subheader("Combined Comparison (Original | Denoised | Enhanced)")
if st.button("Prepare & Download Combined Comparison"):
    try:
        # choose reasonably large height but not huge
        target_h = min(640, orig_pil.height, denoised_pil.height, enhanced_pil.height)
        def resize_keep(p, th):
            w,h = p.size
            return p.resize((int(w*(th/h)), th))
        a = resize_keep(orig_pil, target_h)
        b = resize_keep(denoised_pil, target_h)
        c = resize_keep(enhanced_pil, target_h)
        total_w = a.width + b.width + c.width
        comp = Image.new('RGB', (total_w, target_h))
        comp.paste(a, (0,0)); comp.paste(b, (a.width,0)); comp.paste(c, (a.width+b.width,0))
        comp_bytes = image_to_bytes(comp, fmt="JPEG")
        st.download_button("Download Comparison JPG", data=comp_bytes, file_name="comparison.jpg", mime="image/jpeg")
        st.success("Comparison ready. Click the download button above.")
    except Exception as e:
        st.error("Failed to create comparison: " + str(e))

# Show parameter summary
with st.expander("Parameters used (click to view)"):
    st.write({
        "Auto-adjust": auto_adj,
        "denoise_on": denoise_on,
        "h": h,
        "hColor": hColor,
        "CLAHE_clip": clip,
        "CLAHE_tile": tile,
        "gamma": gamma,
        "sharpen_strength": sharp,
        "bilateral": bilateral
    })
