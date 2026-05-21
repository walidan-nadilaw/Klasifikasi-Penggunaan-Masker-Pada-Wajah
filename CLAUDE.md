# 📋 Rencana Pengembangan Project
## Klasifikasi Penggunaan Masker pada Wajah (Topik 5)

**Mata Kuliah:** Pengolahan Citra Digital (PCD)  
**Durasi:** 30 Hari (12 Mei 2026 – 10 Juni 2026)  
**Jumlah Anggota:** 5 Orang

---

## 1. Deskripsi Project

Membangun sistem klasifikasi citra wajah untuk mendeteksi kondisi penggunaan masker dengan **3 kelas**:

| Kelas | Deskripsi |
|-------|-----------|
| **With Mask** | Wajah memakai masker dengan benar |
| **Without Mask** | Wajah tidak memakai masker |
| **Mask Worn Incorrectly** | Wajah memakai masker dengan tidak tepat |

> [!IMPORTANT]
> Fokus utama project adalah **pemahaman proses**, bukan hanya akurasi akhir. Setiap keputusan dalam pipeline harus memiliki alasan yang jelas.

---

## 2. Dataset yang Digunakan

**Dataset:** [Face Mask 12K Images Dataset by ashishjangra27](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) — Kaggle

| Properti | Detail |
|----------|--------|
| **Total Gambar** | ~12.000 gambar (sudah di-crop area wajah) |
| **Format Gambar** | JPEG |
| **Format Anotasi** | Folder-based (label = nama subfolder) |
| **Split Bawaan** | ✅ Sudah tersedia: `Train/`, `Test/`, `Validation/` |
| **Jumlah Kelas** | 3 kelas |

### Struktur Folder Dataset

```
face-mask-12k-images-dataset/
├── Train/
│   ├── WithMask/         # ~5.000 gambar
│   ├── WithoutMask/      # ~5.000 gambar
│   └── MaskWornIncorrect/
├── Test/
│   ├── WithMask/
│   ├── WithoutMask/
│   └── MaskWornIncorrect/
└── Validation/
    ├── WithMask/
    ├── WithoutMask/
    └── MaskWornIncorrect/
```

> [!NOTE]
> Karena dataset sudah ter-split dan gambar sudah berupa **crop wajah**, tahap face detection/segmentation manual tidak diperlukan. Fokus pipeline beralih ke enhancement, ekstraksi fitur, dan klasifikasi.

---

## 3. Pipeline Teknis

```mermaid
flowchart TD
    A["Load Dataset\nTrain · Validation · Test"] --> B["EDA\nDistribusi kelas · Visualisasi sampel"]
    B --> C["Verifikasi ROI\nFilter gambar kualitas rendah"]
    C --> D["Praproses\nResize 224x224 · Normalisasi 0-1\nAugmentasi pada Train"]

    D --> E{Skenario A}
    E -->|tanpa enhancement| F["Baseline\nCitra asli"]
    E -->|dengan enhancement| G["Enhancement\nCLAHE · Denoising\nSharpening · Gamma Correction"]

    F --> H{Skenario C}
    G --> H

    H -->|Jalur Klasik| I["Ekstraksi Fitur\nHOG · LBP · Color Histogram · GLCM"]
    H -->|Jalur CNN| J["MobileNetV2\nTransfer Learning\nEnd-to-end"]

    I --> K["PCA\nReduksi Dimensi"]
    K --> L["Klasifikasi ML\nSVM · Random Forest · KNN"]

    L --> M["Evaluasi\nAccuracy · Precision · Recall · F1\nConfusion Matrix"]
    J --> M

    M --> N["Analisis Perbandingan\nSkenario A · B · C"]
    N --> O["Laporan & Presentasi"]

    style A fill:#1e3a5f,stroke:#4a9eff,color:#fff
    style B fill:#1e3a5f,stroke:#4a9eff,color:#fff
    style C fill:#1e3a5f,stroke:#4a9eff,color:#fff
    style D fill:#1e3a5f,stroke:#4a9eff,color:#fff
    style E fill:#2d2d2d,stroke:#aaa,color:#fff
    style F fill:#2d4a1e,stroke:#6abf40,color:#fff
    style G fill:#2d4a1e,stroke:#6abf40,color:#fff
    style H fill:#2d2d2d,stroke:#aaa,color:#fff
    style I fill:#5a1e5f,stroke:#c040bf,color:#fff
    style J fill:#5a1e5f,stroke:#c040bf,color:#fff
    style K fill:#5a1e5f,stroke:#c040bf,color:#fff
    style L fill:#5a1e5f,stroke:#c040bf,color:#fff
    style M fill:#5f3a1e,stroke:#ff8c4a,color:#fff
    style N fill:#5f3a1e,stroke:#ff8c4a,color:#fff
    style O fill:#5f3a1e,stroke:#ff8c4a,color:#fff
```

### Detail Setiap Tahap Pipeline

#### 3.1 Akuisisi & Pemahaman Dataset
- Download dataset dari Kaggle (`ashishjangra27/face-mask-12k-images-dataset`)
- Load gambar langsung dari struktur folder `Train/WithMask/`, `Train/WithoutMask/`, dst.
- Eksplorasi distribusi kelas per split (Train, Test, Validation)
- Visualisasi sampel citra per kelas
- Analisis resolusi, ukuran, dan variasi pencahayaan tiap kelas
- Verifikasi integritas data: gambar rusak, duplikat, atau label salah

#### 3.2 Praproses Data
- **Resize** semua citra ke ukuran seragam (misal: 224×224 atau 128×128)
- **Normalisasi** nilai piksel ke rentang [0, 1]
- **Augmentasi** data: rotasi, flip horizontal, brightness adjustment — diterapkan hanya pada data **Train**
- **Split sudah tersedia** dari dataset (Train / Validation / Test) — tidak perlu split manual

#### 3.3 Enhancement / Restoration Citra
- **Histogram Equalization** (CLAHE) untuk memperbaiki kontras
- **Gaussian Blur / Median Filter** untuk denoising
- **Sharpening** menggunakan kernel konvolusi
- **Koreksi pencahayaan** dengan gamma correction
- ⚠️ **Wajib:** Bandingkan hasil **dengan enhancement vs tanpa enhancement**

#### 3.4 Verifikasi & Penyeragaman ROI Wajah
- Gambar dalam dataset **sudah berupa crop wajah** — tahap deteksi wajah dari foto natural tidak diperlukan
- **Verifikasi ROI:** pastikan semua gambar benar-benar menampilkan area wajah (bukan latar belakang atau objek lain)
- **Padding/centering** jika rasio aspek gambar tidak seragam sebelum resize
- Identifikasi dan tandai gambar yang kualitasnya rendah (sangat blur, oklusi ekstrem, dll.)

#### 3.5 Ekstraksi Fitur
Dua pendekatan yang akan dibandingkan:

| Pendekatan | Fitur | Library |
|------------|-------|---------|
| **Fitur Klasik (Manual)** | HOG, LBP, Color Histogram, GLCM | scikit-image, OpenCV |
| **Fitur Otomatis (CNN)** | Feature maps dari convolutional layers | TensorFlow/Keras, PyTorch |

- **HOG (Histogram of Oriented Gradients)** — menangkap bentuk/tepi
- **LBP (Local Binary Pattern)** — menangkap tekstur
- **Color Histogram** — distribusi warna pada ROI
- **GLCM** — statistik tekstur (contrast, energy, correlation, homogeneity)

#### 3.6 Pemodelan Klasifikasi
Dua pendekatan yang akan dibandingkan:

| Metode | Model | Keterangan |
|--------|-------|------------|
| **ML Klasik** | SVM, Random Forest, KNN + PCA | Menggunakan fitur manual; PCA wajib sebelum KNN untuk reduksi dimensi |
| **Deep Learning** | CNN (custom / MobileNetV2) | End-to-end learning; MobileNetV2 cukup untuk 3 kelas |

#### 3.7 Evaluasi Performa
- **Metrik:** Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix** untuk analisis kesalahan per kelas
- **Classification Report** detail per kelas
- **ROC Curve** (opsional)

#### 3.8 Analisis Hasil
- Perbandingan skenario (lihat bagian **Komponen Analisis**)
- Identifikasi kelemahan sistem
- Saran pengembangan lanjutan

---

## 4. Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Bahasa Pemrograman | Python 3.10+ |
| Image Processing | OpenCV, scikit-image, Pillow |
| Feature Extraction | scikit-image (HOG, LBP, GLCM) |
| Machine Learning | scikit-learn (SVM, RF, KNN) |
| Deep Learning | TensorFlow/Keras atau PyTorch |
| Visualisasi | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |
| Version Control | Git + GitHub |

---

## 5. Timeline 30 Hari (Detail)

### 🗓️ Overview Mingguan

```mermaid
gantt
    title Timeline Project Klasifikasi Masker (30 Hari)
    dateFormat  YYYY-MM-DD
    axisFormat  %d %b

    section Minggu 1 (Riset & Setup)
    Pemilihan topik & studi literatur     :m1a, 2026-05-12, 3d
    Download & eksplorasi dataset         :m1b, 2026-05-14, 2d
    Perancangan pipeline & setup env      :m1c, 2026-05-16, 2d

    section Minggu 2 (Praproses & Enhancement)
    Praproses data (resize, normalisasi)  :m2a, 2026-05-19, 2d
    Enhancement & restoration citra       :m2b, 2026-05-21, 2d
    Deteksi wajah & segmentasi            :m2c, 2026-05-23, 2d

    section Minggu 3 (Fitur & Klasifikasi)
    Ekstraksi fitur klasik                :m3a, 2026-05-26, 2d
    Model ML klasik (SVM, RF, KNN)        :m3b, 2026-05-28, 2d
    Model CNN & evaluasi awal             :m3c, 2026-05-30, 2d

    section Minggu 4 (Analisis & Laporan)
    Analisis perbandingan skenario        :m4a, 2026-06-02, 2d
    Penyusunan laporan                    :m4b, 2026-06-04, 3d
    Slide presentasi & demo               :m4c, 2026-06-07, 3d
    Finalisasi & review                   :m4d, 2026-06-10, 1d
```

---

### 📅 Minggu 1: Riset, Dataset & Setup (12–18 Mei 2026)

> **Tujuan:** Memahami topik, menyiapkan dataset, dan merancang pipeline

| Hari | Tanggal | Kegiatan | PIC (Saran) | Deliverable |
|------|---------|----------|-------------|-------------|
| 1 | Sel, 12 Mei | Diskusi topik, studi literatur tentang face mask detection | Semua | Daftar referensi paper/artikel |
| 2 | Rab, 13 Mei | Review dataset Kaggle, pilih dataset, download | Anggota 1-2 | Dataset terdownload |
| 3 | Kam, 14 Mei | Eksplorasi dataset: distribusi kelas, visualisasi sampel | Anggota 3-4 | Notebook EDA (Exploratory Data Analysis) |
| 4 | Jum, 15 Mei | Analisis karakteristik citra (resolusi, noise, pencahayaan) | Anggota 5 | Laporan karakteristik dataset |
| 5 | Sab, 16 Mei | Setup environment Python, install dependencies | Semua | `requirements.txt`, venv ready |
| 6 | Min, 17 Mei | Perancangan pipeline lengkap, pembagian modul kode | Semua | Dokumen desain pipeline |
| 7 | Sen, 18 Mei | Setup repository GitHub, struktur folder project | Anggota 1 | Repo GitHub ready |

> [!NOTE]
> **Milestone Minggu 1:** Dataset sudah terdownload, EDA selesai, pipeline sudah dirancang, dan environment development sudah ready.

---

### 📅 Minggu 2: Praproses, Enhancement & Segmentasi (19–25 Mei 2026)

> **Tujuan:** Implementasi praproses, enhancement citra, dan deteksi/segmentasi wajah

| Hari | Tanggal | Kegiatan | PIC (Saran) | Deliverable |
|------|---------|----------|-------------|-------------|
| 8 | Sel, 19 Mei | Implementasi resize & normalisasi citra | Anggota 1-2 | Modul `preprocessing.py` |
| 9 | Rab, 20 Mei | Implementasi data augmentation (rotasi, flip, brightness) | Anggota 3 | Modul `augmentation.py` |
| 10 | Kam, 21 Mei | Implementasi CLAHE, histogram equalization | Anggota 4-5 | Modul `enhancement.py` |
| 11 | Jum, 22 Mei | Implementasi denoising (Gaussian, Median filter) + sharpening | Anggota 4-5 | Modul `enhancement.py` (lanjutan) |
| 12 | Sab, 23 Mei | Verifikasi ROI: cek semua gambar valid (wajah terdeteksi, tidak blur ekstrem) | Anggota 1-2 | Modul `face_verification.py` |
| 13 | Min, 24 Mei | Padding/centering ROI, handling gambar dengan rasio tidak seragam | Anggota 1-2 | Modul `segmentation.py` |
| 14 | Sen, 25 Mei | Visualisasi hasil enhancement, before/after comparison | Anggota 3 | Notebook perbandingan |

> [!NOTE]
> **Milestone Minggu 2:** Semua modul praproses & enhancement selesai. Verifikasi ROI selesai (gambar tidak valid terfilter). Visualisasi before/after enhancement tersedia.

---

### 📅 Minggu 3: Ekstraksi Fitur & Klasifikasi (26 Mei – 1 Juni 2026)

> **Tujuan:** Ekstraksi fitur, training model ML klasik & CNN, evaluasi awal

| Hari | Tanggal | Kegiatan | PIC (Saran) | Deliverable |
|------|---------|----------|-------------|-------------|
| 15 | Sel, 26 Mei | Ekstraksi fitur HOG & LBP | Anggota 3-4 | Modul `feature_extraction.py` |
| 16 | Rab, 27 Mei | Ekstraksi fitur Color Histogram & GLCM | Anggota 5 | Modul `feature_extraction.py` (lanjutan) |
| 17 | Kam, 28 Mei | Training model SVM & Random Forest | Anggota 1-2 | Modul `ml_classifier.py` |
| 18 | Jum, 29 Mei | Training model KNN, tuning hyperparameter ML | Anggota 1-2 | Hasil tuning, best params |
| 19 | Sab, 30 Mei | Arsitektur CNN (custom atau transfer learning MobileNetV2) | Anggota 3-4 | Modul `cnn_classifier.py` |
| 20 | Min, 31 Mei | Training CNN, monitoring loss & accuracy | Anggota 3-4 | Model trained, learning curves |
| 21 | Sen, 1 Jun | Evaluasi awal semua model: accuracy, confusion matrix | Anggota 5 | Notebook evaluasi awal |

> [!NOTE]
> **Milestone Minggu 3:** Minimal 2 model ML klasik dan 1 model CNN sudah di-training. Evaluasi awal sudah ada (accuracy, confusion matrix).

---

### 📅 Minggu 4: Analisis, Laporan & Presentasi (2–10 Juni 2026)

> **Tujuan:** Analisis mendalam, penulisan laporan, pembuatan presentasi, finalisasi

| Hari | Tanggal | Kegiatan | PIC (Saran) | Deliverable |
|------|---------|----------|-------------|-------------|
| 22 | Sel, 2 Jun | Analisis: dengan enhancement vs tanpa enhancement | Anggota 4-5 | Tabel perbandingan |
| 23 | Rab, 3 Jun | Analisis: fitur klasik vs CNN, ML klasik vs deep learning | Anggota 3-4 | Grafik perbandingan |
| 24 | Kam, 4 Jun | Analisis error: kasus misklasifikasi, pengaruh pose & oklusi | Anggota 1-2 | Analisis kesalahan |
| 25 | Jum, 5 Jun | Mulai penulisan laporan: Bab 1-3 (Pendahuluan, Dataset, Metode) | Anggota 1-3 | Draft laporan |
| 26 | Sab, 6 Jun | Penulisan laporan: Bab 4-5 (Implementasi, Hasil & Analisis) | Anggota 4-5 | Draft laporan (lanjutan) |
| 27 | Min, 7 Jun | Penulisan laporan: Bab 6 (Kesimpulan & Saran) + revisi | Semua | Laporan final draft |
| 28 | Sen, 8 Jun | Pembuatan slide presentasi | Anggota 1-3 | Slide presentasi |
| 29 | Sel, 9 Jun | Persiapan demo program, rehearsal presentasi | Semua | Demo program siap |
| 30 | Rab, 10 Jun | **Finalisasi:** review akhir semua deliverable | Semua | ✅ Semua deliverable selesai |

> [!NOTE]
> **Milestone Minggu 4:** Laporan final, slide presentasi, dan demo program selesai. Semua analisis perbandingan skenario sudah dibahas.

---

## 6. Komponen Analisis Wajib

Berikut analisis yang **harus ada** dalam laporan:

| No | Komponen Analisis | Deskripsi |
|----|-------------------|-----------|
| 1 | Alasan pemilihan topik & dataset | Mengapa memilih Topik 5 dan dataset tertentu |
| 2 | Karakteristik dataset | Distribusi kelas, ukuran, format, tantangan |
| 3 | Masalah citra | Noise, blur, pencahayaan, variasi pose |
| 4 | Enhancement/restoration | Teknik yang digunakan dan alasannya |
| 5 | Fitur yang diekstraksi | HOG, LBP, GLCM, color histogram — alasan pemilihan |
| 6 | Metode klasifikasi | SVM, RF, KNN, CNN — alasan pemilihan |
| 7 | Hasil evaluasi | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| 8 | **Perbandingan skenario** | ⬇️ Lihat tabel di bawah |
| 9 | Kelemahan sistem | Identifikasi kekurangan |
| 10 | Saran pengembangan | Ide improvement di masa depan |

### Skenario Perbandingan Wajib

| Skenario | Variabel | Output yang Diharapkan |
|----------|----------|----------------------|
| A | Dengan Enhancement vs Tanpa Enhancement | Tabel akurasi, grafik perbandingan |
| B | Fitur Warna vs Fitur Tekstur | Analisis fitur mana yang lebih diskriminatif |
| C | Machine Learning Klasik vs CNN | Perbandingan performa SVM/RF vs CNN |

---

## 7. Struktur Folder Project

```
TugasAkhir - Klasifikasi masker/
├── data/
│   ├── raw/                    # Dataset asli dari Kaggle (struktur Train/Test/Validation)
│   │   ├── Train/
│   │   │   ├── WithMask/
│   │   │   ├── WithoutMask/
│   │   │   └── MaskWornIncorrect/
│   │   ├── Test/
│   │   │   ├── WithMask/
│   │   │   ├── WithoutMask/
│   │   │   └── MaskWornIncorrect/
│   │   └── Validation/
│   │       ├── WithMask/
│   │       ├── WithoutMask/
│   │       └── MaskWornIncorrect/
│   ├── processed/              # Dataset setelah praproses (resize, normalisasi)
│   └── augmented/              # Dataset Train setelah augmentasi
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Praproses & Enhancement
│   ├── 03_feature_extraction.ipynb  # Ekstraksi Fitur
│   ├── 04_ml_classification.ipynb   # ML Klasik
│   ├── 05_cnn_classification.ipynb  # CNN / Deep Learning
│   └── 06_evaluation.ipynb     # Evaluasi & Perbandingan
├── src/
│   ├── preprocessing.py        # Modul praproses
│   ├── augmentation.py         # Modul augmentasi
│   ├── enhancement.py          # Modul enhancement/restoration
│   ├── face_detection.py       # Modul deteksi wajah
│   ├── segmentation.py         # Modul segmentasi/cropping
│   ├── feature_extraction.py   # Modul ekstraksi fitur
│   ├── ml_classifier.py        # Modul klasifikasi ML
│   ├── cnn_classifier.py       # Modul klasifikasi CNN
│   └── utils.py                # Fungsi utilitas umum
├── models/                     # Model yang sudah di-training
├── results/
│   ├── figures/                # Grafik & visualisasi
│   └── metrics/                # Metrik evaluasi (CSV/JSON)
├── docs/
│   ├── laporan.pdf             # Laporan final
│   └── presentasi.pptx         # Slide presentasi
├── requirements.txt            # Dependencies Python
├── README.md                   # Dokumentasi project
├── .gitignore
└── LICENSE
```

---

## 8. Deliverables (Luaran)

| No | Deliverable | Format | Deadline |
|----|-------------|--------|----------|
| 1 | Laporan Project | PDF | 9 Juni 2026 |
| 2 | Kode Program Lengkap | Python (.py + .ipynb) | 9 Juni 2026 |
| 3 | Dataset / Link Sumber Dataset | Folder / URL | 14 Mei 2026 |
| 4 | Slide Presentasi | PPTX / PDF | 9 Juni 2026 |
| 5 | Demo Hasil Program | Notebook / Script | 10 Juni 2026 |

---

## 9. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Kelas tidak seimbang (terutama jika ada kelas kecil) | Akurasi bias ke kelas mayoritas | Cek distribusi saat EDA; terapkan augmentasi dan class weighting jika diperlukan |
| Gambar di dataset ada yang blur/kualitas rendah | Fitur tidak terekstrak dengan baik | Filter gambar buruk saat verifikasi ROI (langkah 3.4) |
| Training CNN terlalu lama | Keterlambatan jadwal | Gunakan transfer learning (MobileNetV2), reduce image size ke 128×128 |
| Overfitting pada model | Performa test buruk | Regularization, dropout, early stopping, augmentasi |
| Pipeline loading data lambat (12K gambar) | Proses training melambat | Gunakan `ImageDataGenerator` dengan `flow_from_directory` atau `tf.data` |
| Anggota kelompok tidak aktif | Beban kerja tidak merata | Daily check-in, pembagian PIC yang jelas |

> [!WARNING]
> **Jangan copy-paste kode dari Kaggle Notebook atau GitHub!** Ini melanggar ketentuan project dan akan berpengaruh pada penilaian. Tulis kode sendiri dengan memahami setiap baris.

---

## 10. Checklist Progress

- [ ] **Minggu 1:** Dataset ready, EDA selesai, pipeline dirancang
- [ ] **Minggu 2:** Praproses, enhancement, dan face detection selesai
- [ ] **Minggu 3:** Model ML & CNN trained, evaluasi awal tersedia
- [ ] **Minggu 4:** Analisis lengkap, laporan, presentasi, dan demo siap

---

> [!IMPORTANT]
> **Reminder:** Orisinalitas kode dan kedalaman analisis adalah komponen penting penilaian. Pastikan setiap keputusan dalam pipeline memiliki **justifikasi yang kuat** dan didokumentasikan dengan baik dalam laporan.
