import cv2
import numpy as np
import pywt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from config import *

from skimage.feature import hog

def extract_canny(img, return_viz=False):
    # 1. Fokus pada area bawah wajah (mulai dari 40% tinggi gambar ke bawah)
    h, w = img.shape
    start_h = int(h * 0.4)
    lower_half = img[start_h:, :]
    
    # 2. Deteksi tepi menggunakan Canny
    edges = cv2.Canny(lower_half, 100, 200)
    
    # 3. Ekstraksi HOG dari peta tepi (Edge Map)
    if return_viz:
        features, hog_image = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
        return features, hog_image, edges
    else:
        features = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features

def extract_dwt(img, return_viz=False):
    # 1. Fokus pada area bawah wajah
    h, w = img.shape
    start_h = int(h * 0.4)
    lower_half = img[start_h:, :]
    
    # 2. DWT Transform
    coeffs = pywt.dwt2(lower_half, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # 3. Menggabungkan energi dari sub-band frekuensi tinggi (Horizontal & Vertikal)
    edge_map = np.sqrt(LH**2 + HL**2)
    edge_map_normalized = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4. Ekstraksi HOG dari peta tepi DWT agar Apple-to-Apple dengan Canny dan mencegah overfitting
    if return_viz:
        features, hog_image = hog(edge_map_normalized, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
        return features, hog_image, edge_map_normalized
    else:
        features = hog(edge_map_normalized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features

def load_and_extract_features(root_dir, require_preprocess=False):
    X_canny_list, X_dwt_list, y_list = [], [], []
    if not Path(root_dir).exists(): return None, None, None

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = Path(root_dir) / cls_name
        if not cls_dir.exists(): continue
            
        images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
        for img_path in tqdm(images, desc=f"Loading {cls_name}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if require_preprocess:
                    from src.preprocess import clahe_scratch, gaussian_blur_scratch, sharpen_scratch
                    img_clahe = clahe_scratch(img)
                    img_blur = gaussian_blur_scratch(img_clahe)
                    img = sharpen_scratch(img_blur)
                    # Kita tidak pakai adaptive thresholding karena bisa merusak tekstur Canny/DWT
                
                img_resized = cv2.resize(img, IMG_SIZE_SVM)
                X_canny_list.append(extract_canny(img_resized))
                X_dwt_list.append(extract_dwt(img_resized))
                y_list.append(cls_idx)
                
    return np.array(X_canny_list, dtype=np.float32), np.array(X_dwt_list, dtype=np.float32), np.array(y_list)

def scale_features(X_train, X_val, X_test):
    if X_train is None: return None, None, None, None
    scaler = StandardScaler(copy=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
