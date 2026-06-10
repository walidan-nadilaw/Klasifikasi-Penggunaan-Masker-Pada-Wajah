import cv2
import numpy as np
import kagglehub
from pathlib import Path
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

def download_data_from_kaggle():
    path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
    kaggle_dir = Path(path) / "Face Mask Dataset"
    if not kaggle_dir.exists():
        kaggle_dir = Path(path)
    return kaggle_dir

def apply_custom_preprocessing(img_array):
    img = img_array.astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    
    return np.expand_dims(img_blur, axis=-1).astype(np.float32)

def mobilenet_preprocessing(img_array):
    img = img_array.astype(np.uint8)
    if img.shape[-1] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_3_channel = np.stack((img,)*3, axis=-1)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(img_3_channel.astype(np.float32))

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
