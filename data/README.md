# Data Folder

Folder ini disimpan untuk struktur proyek, bukan untuk menyimpan dataset mentah di Git.

## Aturan kolaborasi

- Simpan dataset secara lokal di `data/raw/`
- Jangan commit isi `data/raw/` ke repository
- Jika lokasi dataset berbeda, set environment variable `MASK_DATA_DIR`

## Struktur yang diharapkan

```text
data/raw/
├── Train/
│   ├── WithMask/
│   ├── WithoutMask/
│   └── WithMaskIncorrect/
├── Validation/
│   ├── WithMask/
│   ├── WithoutMask/
│   └── WithMaskIncorrect/
└── Test/
    ├── WithMask/
    ├── WithoutMask/
    └── WithMaskIncorrect/
```

## Opsi path dataset

Notebook `notebooks/01_eda.ipynb` akan mencoba dataset dari urutan berikut:

1. Path pada environment variable `MASK_DATA_DIR`
2. `data/raw/` di root repository
3. Folder dataset lain yang punya struktur split `Train/`, `Validation/`, dan `Test/`
