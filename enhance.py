# enhance.py
"""
Safe, self-contained image enhancement utilities.
Provides: enhance_image(...) which the main script imports.
"""

import cv2
import numpy as np

def denoise_color_nlmeans(img_bgr, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """OpenCV colored Non-Local Means denoising."""
    try:
        denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None,
                                                   h, hColor,
                                                   templateWindowSize,
                                                   searchWindowSize)
        return denoised
    except Exception:
        # If OpenCV denoising fails for some reason, return the original image
        return img_bgr.copy()

def apply_clahe_lab(img_bgr, clipLimit=2.0, tileGridSize=(8,8)):
    """Apply CLAHE to the L channel in LAB color space."""
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return img_clahe
    except Exception:
        return img_bgr.copy()

def gamma_correction(img_bgr, gamma=1.0):
    """Simple gamma correction. gamma>1 brightens."""
    if gamma <= 0:
        gamma = 1.0
    invGamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    corrected = cv2.LUT(img_bgr, table)
    return corrected

def sharpen(img_bgr, strength=1.0):
    """Unsharp-like sharpening via kernel."""
    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        # Simple strength scaling: blend between identity and kernel
        if strength != 1.0:
            identity = np.zeros_like(kernel); identity[1,1] = 1.0
            # clamp strength to reasonable bounds
            s = float(max(0.0, min(strength, 5.0)))
            kernel = identity * (1.0 - (s - 1.0) * 0.2) + kernel * (s - 1.0) * 0.2
        sharp = cv2.filter2D(img_bgr, -1, kernel)
        return sharp
    except Exception:
        return img_bgr.copy()

def enhance_image(img_bgr,
                  denoise=True,
                  h=10,
                  hColor=10,
                  clahe_clip=2.0,
                  clahe_tile=8,
                  gamma=1.1,
                  sharpen_strength=1.0,
                  use_skimage_nlmeans=False):
    """
    Full pipeline:
      1) optional denoise (OpenCV colored NLMeans)
      2) CLAHE on luminance (LAB)
      3) gamma correction
      4) optional sharpening
    Returns processed BGR image (uint8).
    """
    if img_bgr is None:
        raise ValueError("enhance_image: img_bgr is None")

    working = img_bgr.copy()

    # 1) Denoise
    if denoise:
        # We don't attempt scikit-image at import time; only if caller requests it
        if use_skimage_nlmeans:
            try:
                # lazy import - only if user asked
                from skimage.restoration import denoise_nl_means, estimate_sigma
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
                sigma_est = np.mean(estimate_sigma(gray, channel_axis=None))
                den = denoise_nl_means(gray, h=0.8 * sigma_est, patch_size=7, patch_distance=11, channel_axis=None, fast_mode=True)
                den_u8 = (np.clip(den, 0.0, 1.0) * 255).astype('uint8')
                working = cv2.cvtColor(den_u8, cv2.COLOR_GRAY2BGR)
            except Exception:
                # fallback to OpenCV method
                working = denoise_color_nlmeans(working, h=h, hColor=hColor)
        else:
            working = denoise_color_nlmeans(working, h=h, hColor=hColor)

    # 2) CLAHE
    working = apply_clahe_lab(working, clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))

    # 3) Gamma
    working = gamma_correction(working, gamma=gamma)

    # 4) Sharpen
    if sharpen_strength and sharpen_strength > 0:
        working = sharpen(working, strength=sharpen_strength)

    return working
