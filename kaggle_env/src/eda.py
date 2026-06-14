from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import cv2

import sys
sys.path.append('..')
from config import CLASSES

RANDOM_SEED = 42

def image_stats(path: Path) -> dict:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        gray = img.convert("L")
        width, height = img.size
        rgb_stat = ImageStat.Stat(rgb)
        gray_stat = ImageStat.Stat(gray)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "file_size_kb": path.stat().st_size / 1024,
        "mean_intensity": gray_stat.mean[0],
        "std_intensity": gray_stat.stddev[0],
        "r_mean": rgb_stat.mean[0],
        "g_mean": rgb_stat.mean[1],
        "b_mean": rgb_stat.mean[2],
    }

def rgb_to_hsv_scratch(img_rgb):
    """
    Konversi RGB ke HSV dari scratch menggunakan formula matematika.
    Input: numpy array image RGB format uint8 (0-255).
    Output: numpy array image HSV format uint8 (H: 0-179, S: 0-255, V: 0-255).
    """
    img = img_rgb.astype(np.float32) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    Cmax = np.max(img, axis=2)
    Cmin = np.min(img, axis=2)
    delta = Cmax - Cmin
    
    # Hitung Hue
    H = np.zeros_like(Cmax)
    
    mask_r = (Cmax == R) & (delta != 0)
    mask_g = (Cmax == G) & (delta != 0)
    mask_b = (Cmax == B) & (delta != 0)
    
    H[mask_r] = 60 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)
    
    H[H < 0] += 360
    
    # Hitung Saturation
    S = np.zeros_like(Cmax)
    mask_cmax = Cmax != 0
    S[mask_cmax] = delta[mask_cmax] / Cmax[mask_cmax]
    
    # Hitung Value
    V = Cmax
    
    # Format OpenCV uint8
    H_out = np.clip(np.round(H / 2), 0, 179).astype(np.uint8)
    S_out = np.clip(np.round(S * 255), 0, 255).astype(np.uint8)
    V_out = np.clip(np.round(V * 255), 0, 255).astype(np.uint8)
    
    return np.stack([H_out, S_out, V_out], axis=2)

def rgb_to_lab_scratch(img_rgb):
    """
    Konversi RGB ke LAB dari scratch menggunakan formula matematika.
    Input: numpy array image RGB format uint8 (0-255).
    Output: numpy array image LAB format uint8.
    """
    img = img_rgb.astype(np.float32) / 255.0
    
    # Inverse sRGB gamma correction
    mask = img > 0.04045
    img_lin = np.zeros_like(img)
    img_lin[mask] = np.power((img[mask] + 0.055) / 1.055, 2.4)
    img_lin[~mask] = img[~mask] / 12.92
    
    # RGB to XYZ matrix (Illuminant D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    XYZ = np.dot(img_lin, matrix.T)
    
    # Normalize by D65 white point
    Xn, Yn, Zn = 0.950456, 1.000000, 1.088754
    x = XYZ[:,:,0] / Xn
    y = XYZ[:,:,1] / Yn
    z = XYZ[:,:,2] / Zn
    
    # f(t) function
    def f(t):
        mask_t = t > 0.008856
        res = np.zeros_like(t)
        res[mask_t] = np.cbrt(t[mask_t])
        res[~mask_t] = (7.787 * t[~mask_t]) + (16 / 116)
        return res
    
    fx, fy, fz = f(x), f(y), f(z)
    
    # L, a, b computation
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    # Convert to OpenCV uint8 range
    L_out = np.clip(np.round(L * 255 / 100), 0, 255).astype(np.uint8)
    a_out = np.clip(np.round(a + 128), 0, 255).astype(np.uint8)
    b_out = np.clip(np.round(b + 128), 0, 255).astype(np.uint8)
    
    return np.stack([L_out, a_out, b_out], axis=2)

def convert_color_spaces(path: Path):
    """
    Membaca citra dari path dan mengonversinya ke RGB, Grayscale, HSV, dan LAB.
    Menggunakan konversi dari scratch.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Gambar tidak dapat dibaca di {path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_hsv = rgb_to_hsv_scratch(img_rgb)
    img_lab = rgb_to_lab_scratch(img_rgb)
    
    return {
        "rgb": img_rgb,
        "gray": img_gray,
        "hsv": img_hsv,
        "lab": img_lab
    }

def file_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sample_paths(frame: pd.DataFrame, split: str, label: str, n: int = 3) -> pd.DataFrame:
    subset = frame[(frame["split"] == split) & (frame["label"] == label)]
    if subset.empty:
        return subset
    return subset.sample(n=min(n, len(subset)), random_state=RANDOM_SEED)

def show_sample_grid(frame: pd.DataFrame, split: str, n: int = 3):
    labels = CLASSES
    fig, axes = plt.subplots(len(labels), n, figsize=(4 * n, 4 * len(labels)))
    if len(labels) == 1:
        axes = np.array([axes])

    for i, label in enumerate(labels):
        subset = sample_paths(frame, split, label, n)
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(subset):
                row = subset.iloc[j]
                ax.imshow(Image.open(row["path"]))
                ax.set_title(label + "\n" + row["filename"], fontsize=9)
    plt.suptitle(f"Contoh sampel {split}", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()
