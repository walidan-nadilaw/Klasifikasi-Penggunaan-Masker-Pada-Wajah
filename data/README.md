# Data Folder

Folder ini digunakan sebagai wadah/pusat direktori data untuk eksperimen proyek. **Harap jangan menyimpan atau men-commit dataset mentah (.png/.jpg) maupun file `.npz` ke dalam repository Git.** 

## Mekanisme Loading Dataset (Baru)

Sistem loading dataset telah diperbarui untuk **tidak lagi mengharuskan download manual**. 
Mulai dari Notebook `01_eda.ipynb`, dataset mentah diambil secara otomatis menggunakan library `kagglehub` melalui perintah:
```python
import kagglehub
path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
```
`kagglehub` akan otomatis mengunduh dataset dan menaruhnya di cache lokal bawaan sistem (tidak di repository ini), namun struktur `Train/`, `Validation/`, dan `Test/` miliknya langsung terdeteksi.

## Struktur Folder `data/` yang Dihasilkan

Ketika kamu mengeksekusi urutan *notebook*, folder `data/` ini akan otomatis terisi dengan file/folder *generate* berikut:

```text
data/
├── results/                     # Ter-generate setelah run Notebook 02_preprocesses.ipynb
│   ├── augmented/               
│   │   └── Train/               # Dataset Train yang telah melewati pipeline praproses & augmentasi
│   │       ├── WithMask/
│   │       ├── WithoutMask/
│   │       └── MaskWornIncorrect/
│   └── processed/               
│       ├── Validation/          # Dataset Validation yang HANYA di-praproses (tanpa augmentasi)
│       └── Test/                # Dataset Test yang HANYA di-praproses
│
└── features/                    # Ter-generate setelah run Notebook 03_featureEngineering.ipynb
    ├── canny_aug_features.npz   # Ekstraksi fitur Canny untuk Train (Augmented)
    ├── canny_unaug_features.npz # Ekstraksi fitur Canny untuk Train (Raw)
    ├── dwt_aug_features.npz     # Ekstraksi fitur DWT untuk Train (Augmented)
    └── dwt_unaug_features.npz   # Ekstraksi fitur DWT untuk Train (Raw)
```

## Aturan Kolaborasi

- Semua folder di dalam `data/` selain `README.md` ini akan otomatis di-ignore oleh git (sesuai settingan di `.gitignore`).
- Jika kamu me-run di Kaggle Notebook, mekanisme baca-tulis ini akan otomatis beradaptasi (`04_modelling.ipynb` akan membaca `kagglehub` jika tidak menemukan versi augmentasi lokal).
- Tidak perlu mengubah environment variable (`MASK_DATA_DIR`) lagi. Cukup jalankan Notebook dari urutan 01 sampai 05 secara sekuensial.
