from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

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
