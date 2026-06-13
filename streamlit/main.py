import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# Menambahkan folder kaggle_env ke sys.path menggunakan absolute path yang pasti
current_file_path = Path(os.path.abspath(__file__))
PROJECT_ROOT = current_file_path.parent.parent
KAGGLE_ENV_ROOT = PROJECT_ROOT / "kaggle_env"
sys.path.insert(0, str(KAGGLE_ENV_ROOT))

# Mengimpor modul buatan dari kaggle_env
try:
    from src.preprocess import clahe_scratch, gaussian_blur_scratch, sharpen_scratch, mobilenet_prep_enhance, mobilenet_prep_no_enhance, cnn_prep_enhance, cnn_prep_no_enhance
except ImportError as e:
    st.error(f"Gagal mengimpor library: {e}")

# Setup Halaman
st.set_page_config(page_title="Mask Detection & Live Preprocessing", page_icon="😷", layout="wide")

st.title("😷 Face Mask Detection & Live Preprocessing")
st.write("Demonstrasi klasifikasi penggunaan masker secara live dengan deteksi wajah dan preprocessing.")

# Sidebar Konfigurasi
st.sidebar.header("⚙️ Konfigurasi Sistem")
selected_model_name = st.sidebar.selectbox(
    "Pilih Model Klasifikasi:",
    [
        "MobileNetV2 (Dengan Enhancement)",
        "MobileNetV2 (Tanpa Enhancement)",
        "SVM - Canny (Dengan Enhancement)",
        "SVM - Canny (Tanpa Enhancement)",
        "SVM - DWT (Dengan Enhancement)",
        "SVM - DWT (Tanpa Enhancement)"
    ]
)

@st.cache_resource
def load_classification_model(model_name):
    """Meload model Deep Learning atau Machine Learning yang sudah ditraining."""
    file_map = {
        "MobileNetV2 (Dengan Enhancement)": "mob_enh.h5",
        "MobileNetV2 (Tanpa Enhancement)": "mob_no_enh.h5",
        "SVM - Canny (Dengan Enhancement)": "svm_c_enh.pkl",
        "SVM - Canny (Tanpa Enhancement)": "svm_c_no_enh.pkl",
        "SVM - DWT (Dengan Enhancement)": "svm_d_enh.pkl",
        "SVM - DWT (Tanpa Enhancement)": "svm_d_no_enh.pkl"
    }
    
    filename = file_map.get(model_name)
    model_path = PROJECT_ROOT / "kaggle_env" / "models" / filename
    
    if not model_path.exists():
        return None
        
    if "MobileNet" in model_name:
        from src.model import build_mobilenet_model
        model = build_mobilenet_model(num_classes=2)
        model.load_weights(str(model_path))
        return model
    else:
        # Load model SVM pakai joblib
        import joblib
        model = joblib.load(str(model_path))
        return model

model = load_classification_model(selected_model_name)

# Setup Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Area Input Gambar (Menggunakan Tabs untuk opsi Upload atau Webcam)
tab1, tab2 = st.tabs(["📁 Upload File", "📸 Gunakan Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    
with tab2:
    camera_file = st.camera_input("Ambil Foto dari Webcam")

# Gunakan file dari webcam jika ada, jika tidak gunakan file upload
active_file = camera_file if camera_file is not None else uploaded_file

if active_fijle is not None:
    # Membaca byte gambar menjadi array numpy
    file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
    img_bgr_full = cv2.imdecode(file_bytes, 1)
    
    # ----------------------------------------------------
    # FASE 0: ENHANCEMENT FULL IMAGE (RGB) UNTUK DETEKSI
    # ----------------------------------------------------
    st.divider()
    st.subheader("🔍 Fase 0: Peningkatan Resolusi & Deteksi Wajah (Full Image)")
    st.write("Mengaplikasikan fungsi *scratch* pada channel *Value* (HSV) agar warna RGB tidak rusak, demi mempermudah kerja *Face Detector*.")
    
    with st.spinner("Meningkatkan kualitas gambar penuh (Enhancement RGB)..."):
        # Konversi ke HSV untuk meng-enhance kecerahannya saja (Channel V)
        hsv_full = cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv_full)
        
        # Aplikasikan fungsi scratch ke channel V
        v_clahe = clahe_scratch(v_chan)
        v_blur = gaussian_blur_scratch(v_clahe)
        v_sharp = sharpen_scratch(v_blur)
        
        # Gabungkan kembali dan kembalikan ke BGR
        hsv_enhanced = cv2.merge([h_chan, s_chan, v_sharp])
        img_bgr_full_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    col_raw1, col_raw2 = st.columns(2)
    col_raw1.image(cv2.cvtColor(img_bgr_full, cv2.COLOR_BGR2RGB), caption="1. Foto Raw (Asli)", use_column_width=True)
    col_raw2.image(cv2.cvtColor(img_bgr_full_enhanced, cv2.COLOR_BGR2RGB), caption="2. Foto Enhanced (RGB)", use_column_width=True)
    
    # ----------------------------------------------------
    # DETEKSI WAJAH (Menggunakan Gambar yang sudah di-Enhance)
    # ----------------------------------------------------
    # Konversi gambar enhanced ke grayscale khusus untuk detector Haar
    gray_enhanced = cv2.cvtColor(img_bgr_full_enhanced, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        st.error("🚨 Wajah tidak terdeteksi bahkan setelah di-enhance! Mohon posisikan wajah Anda lebih jelas ke arah kamera.")
    else:
        # Ambil wajah terbesar (asumsi itu adalah subjek utama)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Tambahkan sedikit padding
        pad = int(w * 0.15)
        y1 = max(0, y - pad)
        y2 = min(img_bgr_full.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_bgr_full.shape[1], x + w + pad)
        
        # Tampilkan box deteksi di UI (pada gambar RGB Enhanced)
        preview_img = img_bgr_full_enhanced.copy()
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        st.success("✅ Wajah berhasil terdeteksi dari foto yang telah di-enhance!")
        col_raw1.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), caption="3. Deteksi Area Wajah (Tracking)", use_column_width=True)
            
        # ----------------------------------------------------
        # FASE 1: PREPROCESSING CROP WAJAH RAW
        # ----------------------------------------------------
        # Sesuai permintaan: Crop dari gambar RAW, lalu di-enhance lagi dari awal
        img_bgr = img_bgr_full[y1:y2, x1:x2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        st.divider()
        st.subheader("🛠️ Fase 1: Live Image Enhancement Wajah (Step-by-Step dari Scratch)")
        
        # Menampilkan grid tahapan
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Tahap 1: Gambar Asli (Cropped)
        col1.image(img_rgb, caption="1. Original (Cropped)", use_column_width=True)
        
        # Tahap 2: Grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        col2.image(img_gray, caption="2. Grayscale", use_column_width=True, channels="GRAY")
        
        # Tahap 3: CLAHE (Scratch)
        img_clahe = clahe_scratch(img_gray)
        col3.image(img_clahe, caption="3. CLAHE (Kontras)", use_column_width=True, channels="GRAY")
        
        # Tahap 4: Gaussian Blur (Scratch)
        img_blur = gaussian_blur_scratch(img_clahe)
        col4.image(img_blur, caption="4. Gaussian Blur (Denoise)", use_column_width=True, channels="GRAY")
        
        # Tahap 5: Sharpening (Scratch)
        img_sharp = sharpen_scratch(img_blur)
        col5.image(img_sharp, caption="5. Sharpening (Tajam)", use_column_width=True, channels="GRAY")
        
        st.divider()
        st.subheader(f"🎯 2. Hasil Prediksi Model: {selected_model_name}")
        
        if model is None:
            st.warning(f"⚠️ Model '{selected_model_name}' belum ditraining atau tidak ditemukan di folder `kaggle_env/models`.")
        else:
            classes = ["WithMask", "WithoutMask"]
            pred_label = None
            confidence = 0.0
            is_error = False
            
            with st.spinner("Menganalisis wajah..."):
                if "MobileNet" in selected_model_name:
                    # PROSES UNTUK MOBILENET
                    img_resized = cv2.resize(img_bgr, (224, 224))
                    if "Dengan Enhancement" in selected_model_name:
                        processed_input = mobilenet_prep_enhance(img_resized)
                    else:
                        processed_input = mobilenet_prep_no_enhance(img_resized)
                        
                    batch_input = np.expand_dims(processed_input, axis=0)
                    pred_probs = model.predict(batch_input)[0]
                    pred_idx = np.argmax(pred_probs)
                    pred_label = classes[pred_idx]
                    confidence = pred_probs[pred_idx] * 100
                else:
                    # PROSES UNTUK SVM
                    import joblib
                    from src.feature_engineering import extract_canny, extract_dwt
                    
                    # 1. Pilih gambar yg mau di-resize
                    if "Tanpa Enhancement" in selected_model_name:
                        img_input = img_gray
                    else:
                        img_input = img_sharp
                        
                    # Target size SVM adalah 224x224
                    img_resized = cv2.resize(img_input, (224, 224))
                    
                    # 2. Ekstrak Fitur
                    if "Canny" in selected_model_name:
                        feature = extract_canny(img_resized)
                        scaler_name = "scaler_c_enh.pkl" if "Dengan" in selected_model_name else "scaler_c_no_enh.pkl"
                    else:
                        feature = extract_dwt(img_resized)
                        scaler_name = "scaler_d_enh.pkl" if "Dengan" in selected_model_name else "scaler_d_no_enh.pkl"
                    
                    # Reshape ke 2D array untuk sklearn
                    feature = feature.reshape(1, -1)
                    
                    # 3. Load Scaler
                    scaler_path = PROJECT_ROOT / "kaggle_env" / "models" / scaler_name
                    if not scaler_path.exists():
                        st.error(f"⚠️ File scaler '{scaler_name}' tidak ditemukan! Anda harus menyimpan scaler dari notebook `03_training.ipynb` menggunakan `joblib.dump(scaler, 'models/{scaler_name}')` sebelum bisa melakukan prediksi live dengan SVM.")
                        is_error = True
                    else:
                        scaler = joblib.load(str(scaler_path))
                        feature_scaled = scaler.transform(feature)
                        
                        # 4. Prediksi SVM
                        pred_idx = model.predict(feature_scaled)[0]
                        pred_label = classes[pred_idx]
                        confidence = 100.0 # SVM standar tidak memgeluarkan probabilitas
            
            if not is_error:
                # Tampilkan Hasil Visual
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    # Tampilkan versi gambar yang dikirim ke model
                    if "Tanpa Enhancement" in selected_model_name:
                        st.image(img_rgb, caption="Input (Tanpa Enhance)", use_column_width=True)
                    else:
                        st.image(img_sharp, caption="Input (Dengan Enhance)", use_column_width=True, channels="GRAY")
                    
                with col_res2:
                    st.markdown(f"### Diagnosis Sistem:")
                    if pred_label == "WithMask":
                        st.success(f"## 😷 TERDETEKSI MEMAKAI MASKER")
                    else:
                        st.error(f"## 🚨 TIDAK MEMAKAI MASKER")
                        
                    if "MobileNet" in selected_model_name:
                        st.progress(int(confidence))
                        st.write(f"Tingkat Keyakinan Model (Confidence): **{confidence:.2f}%**")
                    else:
                        st.write("*(Catatan: SVM menggunakan margin klasifikasi biner mutlak, sehingga tingkat keyakinan tidak direpresentasikan dalam persen)*")
