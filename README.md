# 🎓 Early Warning System: Prediksi Performa Pembelajaran Siswa Berbasis Explainable AI (XAI)

Dashboard interaktif berbasis web untuk mendeteksi performa pembelajaran siswa secara dini (*Early Warning System*) menggunakan algoritma **Machine Learning** yang dilengkapi dengan **Explainable AI (SHAP)** untuk transparansi dan interpretasi hasil prediksi.

---

## 📌 Gambaran Umum Proyek

Proyek ini merupakan implementasi penelitian skripsi yang bertujuan untuk:
1. Mengidentifikasi siswa yang berisiko mengalami penurunan performa akademik berdasarkan aktivitas pembelajaran pada *Learning Management System* (LMS).
2. Memberikan interpretasi dan visualisasi kontribusi setiap fitur/variabel terhadap hasil prediksi menggunakan metode **SHAP (SHapley Additive exPlanations)**.
3. Membantu tenaga pengajar atau institusi pendidikan dalam mengambil keputusan dan memberikan intervensi secara tepat waktu dan berbasis data.

---

## 🚀 Fitur Utama

- **📊 Dashboard Interaktif (Streamlit):** Antarmuka pengguna yang responsif, modern, dan mudah digunakan.
- **📥 Prediksi Batch (File Upload):** Mendukung unggah berkas `.csv` untuk melakukan prediksi terhadap data banyak siswa sekaligus.
- **🎯 Klasifikasi Risiko:** Mengelompokkan siswa ke dalam tingkat risiko (Tinggi, Sedang, Rendah).
- **🔍 Explainable AI (XAI):**
  - **Global Interpretability:** Menampilkan fitur yang paling berpengaruh secara umum terhadap prediksi model.
  - **Local Interpretability:** Memberikan penjelasan spesifik untuk masing-masing siswa (alasan mengapa seorang siswa masuk ke kategori risiko tertentu).
- **📋 Rekomendasi Intervensi:** Memberikan saran tindakan preventif bagi siswa yang membutuhkan perhatian khusus.
- **💾 Export Laporan:** Mengunduh hasil prediksi dan analisis risiko dalam format `.csv`.

---

## 🛠️ Tech Stack & Dependensi

- **Bahasa Pemrograman:** Python (3.9+)
- **Framework Web Dashboard:** Streamlit
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Explainable AI (XAI):** SHAP
- **Manipulasi & Analisis Data:** Pandas, NumPy
- **Visualisasi Data:** Matplotlib, Plotly

---

## 📁 Struktur Direktori

```text
early-warning-system-ml/
│
├── app.py                      # Aplikasi utama dashboard Streamlit
├── requirements.txt            # Daftar dependensi library Python
├── .gitignore                  # Berkas pengecualian Git
├── README.md                   # Dokumentasi proyek
│
├── models/                     # Model ML & Scaler yang telah dilatih
│   ├── best_model.pkl          # Model prediksi terbaik
│   └── scaler.pkl              # Scaler preprocessing fitur
│
├── data/                       # Dataset pendukung
│   ├── lms_dataset_training.csv      # Dataset acuan pelatihan LMS
│   └── contoh_upload_dashboard.csv   # Template / contoh file CSV untuk diuji
│
├── notebooks/                  # Notebook eksperimen & pelatihan model
│   ├── DATASET.ipynb           # Eksplorasi data (EDA) & preprocessing
│   └── NEW.ipynb               # Pelatihan, tuning, dan evaluasi model
│
└── assets/                     # Diagram alur & visualisasi pendukung
    ├── flowchart_pipeline_ml.png
    ├── flowchart_xai_shap.png
    └── kerangka_berpikir_kuantitatif.png
```

---

## ⚙️ Panduan Instalasi & Menjalankan Proyek

### 1. Prasyarat
Pastikan komputer telah terinstal:
- **Python 3.9** ke atas
- **Git**

### 2. Clone Repository
```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>
```

### 3. Buat dan Aktifkan Virtual Environment (Disarankan)
- **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat default: `http://localhost:8501`.

---

## 📊 Format Input Data (CSV)

Untuk fitur **Batch Prediction**, file CSV yang diunggah diharapkan memiliki kolom fitur aktivitas LMS berikut:
- `jumlah_login` (Total login ke LMS)
- `rata_rata_durasi_sesi` (Durasi rata-rata sesi dalam menit)
- `jumlah_tugas_dikumpulkan` (Jumlah tugas yang telah diserahkan)
- `jumlah_postingan_forum` (Keaktifan berdiskusi di forum LMS)
- `jumlah_percobaan_kuis` (Frekuensi pengerjaan kuis)
- `tingkat_keterlambatan_pengumpulan` (Frekuensi/rasio keterlambatan tugas)
- `rata_rata_nilai_kuis` (Nilai rata-rata kuis siswa)
- `total_waktu_belajar` (Akumulasi waktu belajar dalam jam)

*(File contoh telah disediakan di `data/contoh_upload_dashboard.csv`)*.

---

## 📝 Catatan Lisensi & Akademik

Proyek ini dikembangkan sebagai bagian dari penelitian skripsi/tugas akhir akademik. Kode ini dipublikasikan untuk keperluan edukasi, transparansi penelitian, dan portofolio.
