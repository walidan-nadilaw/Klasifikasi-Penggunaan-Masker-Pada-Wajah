import cv2
import numpy as np
import pywt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from config import *

def extract_canny(img):
    edges = cv2.Canny(img, 100, 200)
    return edges.flatten()

def extract_dwt(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL.flatten()

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
