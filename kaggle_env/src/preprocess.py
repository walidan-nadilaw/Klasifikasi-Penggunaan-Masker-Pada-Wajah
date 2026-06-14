import cv2
import numpy as np
import kagglehub
from pathlib import Path
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

def remove_blank_images(kaggle_dir):
    print("Memeriksa dan menghapus gambar blank hitam...")
    removed = 0
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in kaggle_dir.rglob(ext):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if np.max(img) == 0:
                try:
                    img_path.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Gagal menghapus {img_path}: {e}")
    if removed > 0:
        print(f"Berhasil menghapus {removed} gambar blank hitam dari dataset.")
    else:
        print("Tidak ada gambar blank hitam yang ditemukan (atau sudah dihapus sebelumnya).")

def remove_duplicate_images(kaggle_dir):
    import hashlib
    from collections import defaultdict
    print("Memeriksa dan menghapus gambar duplikat (exact-match)...")
    
    def file_sha1(path):
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1048576)
                if not chunk: break
                h.update(chunk)
        return h.hexdigest()

    hash_dict = defaultdict(list)
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in kaggle_dir.rglob(ext):
            try:
                file_hash = file_sha1(img_path)
                hash_dict[file_hash].append(img_path)
            except Exception:
                continue
                
    removed = 0
    for file_hash, paths in hash_dict.items():
        if len(paths) > 1:
            # Urutkan path agar 'Train' selalu berada di urutan pertama (dipertahankan)
            # Jika ada kebocoran (Train & Validation punya gambar sama), hapus yang di Validation
            paths_sorted = sorted(paths, key=lambda x: 0 if 'Train' in x.parts else 1)
            
            # Hapus semua indeks ke-1 dan seterusnya
            for duplicate_path in paths_sorted[1:]:
                try:
                    duplicate_path.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Gagal menghapus {duplicate_path}: {e}")
                    
    if removed > 0:
        print(f"Berhasil menghapus {removed} gambar duplikat dari dataset.")
    else:
        print("Tidak ada gambar duplikat yang ditemukan.")

def download_data_from_kaggle():
    path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
    original_dir = Path(path) / "Face Mask Dataset"
    if not original_dir.exists():
        original_dir = Path(path)
        
    # Salin ke direktori writable (lokal/Kaggle working dir) agar bisa menghapus gambar rusak
    writable_dir = Path('./dataset_cleaned')
    if not writable_dir.exists():
        print(f"Menyalin dataset dari {original_dir} ke direktori writable ({writable_dir})...")
        shutil.copytree(original_dir, writable_dir)
        
    return writable_dir

def clahe_scratch(img, clip_limit=2.0, grid_size=(8,8)):
    h, w = img.shape
    gh, gw = grid_size
    cell_h, cell_w = h // gh, w // gw
    cdfs = np.zeros((gh, gw, 256), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            r_start, r_end = i * cell_h, min((i+1) * cell_h, h)
            c_start, c_end = j * cell_w, min((j+1) * cell_w, w)
            block = img[r_start:r_end, c_start:c_end]
            hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
            actual_clip = clip_limit * (block.size / 256)
            excess = np.maximum(hist - actual_clip, 0)
            hist = np.minimum(hist, actual_clip)
            hist += excess.sum() / 256
            cdf = np.cumsum(hist)
            if cdf[-1] > 0: cdfs[i, j] = cdf * 255 / cdf[-1]
    r_grid, c_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    i_y, j_x = (r_grid - cell_h / 2) / cell_h, (c_grid - cell_w / 2) / cell_w
    i1, j1 = np.floor(i_y).astype(int), np.floor(j_x).astype(int)
    i2, j2 = i1 + 1, j1 + 1
    p, q = i_y - i1, j_x - j1
    i1_c, j1_c = np.clip(i1, 0, gh-1), np.clip(j1, 0, gw-1)
    i2_c, j2_c = np.clip(i2, 0, gh-1), np.clip(j2, 0, gw-1)
    val = img
    cdf11, cdf12 = cdfs[i1_c, j1_c, val], cdfs[i1_c, j2_c, val]
    cdf21, cdf22 = cdfs[i2_c, j1_c, val], cdfs[i2_c, j2_c, val]
    out = (1-p)*(1-q)*cdf11 + (1-p)*q*cdf12 + p*(1-q)*cdf21 + p*q*cdf22
    return np.clip(out, 0, 255).astype(np.uint8)

def gaussian_blur_scratch(img, size=5, sigma=1.0):
    k = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for i in range(size):
        for j in range(size):
            k[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    k /= np.sum(k)
    padded = np.pad(img, center, mode='reflect').astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(size):
        for j in range(size):
            out += padded[i:i+img.shape[0], j:j+img.shape[1]] * k[i, j]
    return np.clip(out, 0, 255).astype(np.uint8)

def sharpen_scratch(img):
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]], dtype=np.float32)
    padded = np.pad(img, 1, mode='reflect').astype(np.float32)
    out = (padded[:-2, :-2]*kernel[0,0] + padded[:-2, 1:-1]*kernel[0,1] + padded[:-2, 2:]*kernel[0,2] +
           padded[1:-1, :-2]*kernel[1,0] + padded[1:-1, 1:-1]*kernel[1,1] + padded[1:-1, 2:]*kernel[1,2] +
           padded[2:, :-2]*kernel[2,0] + padded[2:, 1:-1]*kernel[2,1] + padded[2:, 2:]*kernel[2,2])
    return np.clip(out, 0, 255).astype(np.uint8)

def adaptive_threshold_scratch(img, block_size=11, C=2):
    pad = block_size // 2
    padded = np.pad(img, pad, mode='reflect').astype(np.float32)
    h, w = img.shape
    integral = np.zeros((padded.shape[0]+1, padded.shape[1]+1), dtype=np.float32)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    i_br = integral[block_size:block_size+h, block_size:block_size+w]
    i_tr = integral[0:h, block_size:block_size+w]
    i_bl = integral[block_size:block_size+h, 0:w]
    i_tl = integral[0:h, 0:w]
    local_mean = (i_br - i_tr - i_bl + i_tl) / (block_size * block_size)
    out = np.zeros_like(img, dtype=np.uint8)
    out[img > (local_mean - C)] = 255
    return out

def apply_custom_preprocessing(img_array):
    img = img_array.astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.squeeze()
        
    img_clahe = clahe_scratch(img_gray)
    img_blur = gaussian_blur_scratch(img_clahe)
    img_sharp = sharpen_scratch(img_blur)
    img_thresh = adaptive_threshold_scratch(img_sharp)
    
    return np.expand_dims(img_thresh, axis=-1).astype(np.float32)

def mobilenet_prep_no_enhance(img_array):
    img = img_array.astype(np.uint8)
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img_gray = img.squeeze()
        img_3_channel = np.stack((img_gray,)*3, axis=-1)
    else:
        img_3_channel = img
        
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(img_3_channel.astype(np.float32))

def mobilenet_prep_enhance(img_array):
    img = img_array.astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.squeeze()
        
    img_clahe = clahe_scratch(img_gray)
    img_blur = gaussian_blur_scratch(img_clahe)
    img_sharp = sharpen_scratch(img_blur)
    
    img_3_channel = np.stack((img_sharp,)*3, axis=-1)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(img_3_channel.astype(np.float32))

def cnn_prep_no_enhance(img_array):
    img = img_array.astype(np.float32)
    # Custom CNN takes 224x224x3 and usually we just scale 0-1 (which ImageDataGenerator(rescale=1./255) handles)
    return img

def cnn_prep_enhance(img_array):
    img = img_array.astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.squeeze()
        
    img_clahe = clahe_scratch(img_gray)
    img_blur = gaussian_blur_scratch(img_clahe)
    img_sharp = sharpen_scratch(img_blur)
    
    img_3_channel = np.stack((img_sharp,)*3, axis=-1)
    return img_3_channel.astype(np.float32)

def setup_generators(kaggle_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=10, width_shift_range=0.2,
        height_shift_range=0.2, zoom_range=0.25, horizontal_flip=True,
        fill_mode='nearest', preprocessing_function=apply_custom_preprocessing
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255, preprocessing_function=apply_custom_preprocessing
    )
    return train_datagen, test_datagen
