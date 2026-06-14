import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / "results"
FEATURES_DIR = DATA_DIR / "features"

# Deteksi otomatis apakah berjalan di Kaggle Notebook
if os.path.exists("/kaggle/working"):
    MODELS_DIR = Path("/kaggle/working/models")
else:
    MODELS_DIR = BASE_DIR / "models"

# Parameters
IMG_SIZE_SVM = (128, 128)
IMG_SIZE_CNN = (224, 224)
BATCH_SIZE = 32
CLASSES = ["WithMask", "WithoutMask"]
