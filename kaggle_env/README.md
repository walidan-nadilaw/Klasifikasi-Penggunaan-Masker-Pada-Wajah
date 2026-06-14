# Panduan Menjalankan Pipeline di Kaggle Notebook

Dokumen ini berisi panduan langkah demi langkah untuk menjalankan eksperimen Face Mask Classification (Skenario SVM dan MobileNetV2) secara komplit di environment Kaggle, tanpa harus mengunduh data ke komputer lokal atau pusing memikirkan masalah *MemoryError*.

Pipeline yang dijalankan di sini menggunakan struktur "On-The-Fly", di mana pemrosesan gambar dan augmentasi dilakukan langsung di dalam memori (RAM), sehingga jauh lebih cepat dan tidak memakan ruang penyimpanan keras.

---

## Tahap 1: Persiapan Modul (Membuat Dataset Kaggle)
Karena kode terpisah menjadi beberapa file modul (`src/` dan `config.py`), kita harus meng-uploadnya terlebih dahulu sebagai **Kaggle Dataset**.

1. Buka [Kaggle](https://www.kaggle.com/) dan login.
2. Di menu kiri, klik **Datasets** -> **New Dataset**.
3. Berikan judul yang mudah diingat (misalnya: `pcd-face-mask-scripts`).
4. Tarik dan lepas (Drag & Drop) file `config.py` beserta seluruh folder `src/` ke area upload. 
   *(Tips: Anda bisa melakukan ZIP pada folder `src` dan file `config.py` terlebih dahulu, Kaggle akan otomatis mengekstraknya saat di-upload).*
5. Klik **Create** dan tunggu prosesnya selesai.

---

## Tahap 2: Persiapan Notebook
Selanjutnya, siapkan arena eksekusinya (Notebook).

1. Buka menu **Notebooks** -> **New Notebook** di Kaggle.
2. Masukkan judul untuk notebook Anda.
3. Di panel kanan (bagian *Input*), klik **+ Add Input**.
4. Pilih tab **Your Datasets** dan tambahkan dataset `pcd-face-mask-scripts` yang baru Anda buat tadi dengan menekan tombol **+**.
5. Buka tab **File** (pojok kiri atas notebook), lalu pilih **Import Notebook**. 
6. Upload file `kaggle_train.ipynb` dari komputer lokal Anda. Kaggle akan memuat semua isi sel yang ada pada file tersebut.

---

## Tahap 3: Konfigurasi Path dan Hardware
Sebelum menekan tombol *Run*, pastikan beberapa konfigurasi vital ini:

1. **Atur Path Dataset Code:**
   Di sel pertama notebook, Anda akan melihat kode yang dikomentari seperti ini:
   ```python
   # import sys
   # sys.path.append('/kaggle/input/datasets/username/namadataset')
   ```
   Hilangkan tanda pagar (`#`) dan **ganti path-nya** sesuai dengan lokasi direktori dataset *scripts* yang Anda upload. 
   *(Tips: Klik tombol "Copy File Path" di dataset Anda pada panel kanan Kaggle untuk mendapatkan path yang akurat).*

2. **Gunakan Akselerasi GPU:**
   Karena kita akan melatih MobileNetV2 (Deep Learning), disarankan menggunakan GPU.
   - Buka panel kanan Notebook, cari menu **Session Options** -> **Accelerator**.
   - Pilih **GPU T4 x2** atau **GPU P100**.

---

## Tahap 4: Eksekusi dan Pengambilan Model (Save & Commit)
Anda dapat menjalankan sel satu-per-satu untuk memastikan semuanya berjalan lancar. 

Namun, jika Anda ingin menjalankan keseluruhan proses dari awal hingga evaluasi Skenario 4 sambil ditinggal minum kopi:
1. Di pojok kanan atas, klik tombol **Save Version**.
2. Pilih tipe penyimpanan: **Save & Run All (Commit)**.
3. Klik **Save**.
   
Kaggle akan mulai menjalankan kode di *background server*. 
Setelah statusnya menjadi **Successful**, buka versi tersebut, lalu navigasikan ke tab **Output**. Anda akan menemukan folder `models` yang berisi file `svm_canny_unaug.pkl`, `svm_canny_aug.pkl`, `mobilenet_unaug.h5`, dan `mobilenet_aug.h5`. 

Klik **Download** untuk menggunakannya di aplikasi atau *deployment* Anda!
