# ==========================================================
# EARLY WARNING SYSTEM
# Student Performance Prediction using Machine Learning
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Early Warning System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Early Warning System")
st.subheader(
    "Prediksi Performa Pembelajaran Siswa Menggunakan "
    "Optimasi Machine Learning Berbasis Explainable AI"
)

st.markdown("---")

# ==========================================================
# BASE PATH & MODEL PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "early warning system" / "best_model.pkl"

SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
if not SCALER_PATH.exists():
    SCALER_PATH = BASE_DIR / "early warning system" / "scaler.pkl"

TRAIN_PATH = BASE_DIR / "data" / "lms_dataset_training.csv"
if not TRAIN_PATH.exists():
    TRAIN_PATH = BASE_DIR / "lms_dataset_training.csv"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


model, scaler = load_model()

# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

@st.cache_data
def load_training():

    return pd.read_csv(TRAIN_PATH)


training_df = load_training()

# ==========================================================
# FEATURE YANG DIGUNAKAN MODEL
# ==========================================================

FEATURES = [

    "jumlah_login",

    "rata_rata_durasi_sesi",

    "jumlah_tugas_dikumpulkan",

    "jumlah_postingan_forum",

    "jumlah_percobaan_kuis",

    "tingkat_keterlambatan_pengumpulan",

    "rata_rata_nilai_kuis",

    "total_waktu_belajar"

]

# ==========================================================
# LABEL
# ==========================================================

if hasattr(model, "classes_"):

    LABELS = list(model.classes_)

else:

    LABELS = [0,1,2]

LABEL_MAP = {

    0: "Rendah",

    1: "Sedang",

    2: "Tinggi"

}

# ==========================================================
# SESSION STATE
# ==========================================================

if "dataset" not in st.session_state:

    st.session_state.dataset = None

if "prediction" not in st.session_state:

    st.session_state.prediction = None

# ==========================================================
# STEP 1
# UPLOAD DATASET
# ==========================================================

st.header("📁 STEP 1 - Upload Dataset")

uploaded_file = st.file_uploader(

    "Upload dataset (.csv)",

    type=["csv"]

)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.session_state.dataset = df

        st.success("Dataset berhasil diupload.")

    except Exception as e:

        st.error(e)

# ==========================================================
# CEK DATASET
# ==========================================================

if st.session_state.dataset is not None:

    df = st.session_state.dataset.copy()

    st.subheader("Informasi Dataset")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Jumlah Data",

        len(df)

    )

    c2.metric(

        "Jumlah Kolom",

        len(df.columns)

    )

    c3.metric(

        "Missing Value",

        int(df.isnull().sum().sum())

    )

    c4.metric(

        "Duplicate",

        int(df.duplicated().sum())

    )

    st.subheader("Preview Dataset")

    st.dataframe(

        df.head(),

        use_container_width=True

    )

    st.markdown("---")
# ==========================================================
# STEP 2
# VALIDASI DATA DAN PREDIKSI
# ==========================================================

if st.session_state.dataset is not None:

    st.header("🤖 STEP 2 - Prediksi Machine Learning")

    df = st.session_state.dataset.copy()

    # ======================================================
    # VALIDASI FITUR
    # ======================================================

    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        st.error(
            "Dataset belum dapat diproses karena masih ada kolom yang belum tersedia."
        )

        st.write("Kolom yang belum ditemukan:")

        st.write(missing_columns)

        st.stop()

    # ======================================================
    # PREPROCESSING
    # ======================================================

    X = df[FEATURES].copy()

    X_scaled = scaler.transform(X)

    st.success("Preprocessing berhasil dilakukan.")

    # ======================================================
    # PREDIKSI
    # ======================================================

    with st.spinner("Melakukan prediksi..."):

        prediction = model.predict(X_scaled)

        probability = model.predict_proba(X_scaled)

    # ======================================================
    # LABEL PREDIKSI
    # ======================================================

    label_prediction = []

    for p in prediction:

        if p in LABEL_MAP:

            label_prediction.append(
                LABEL_MAP[p]
            )

        else:

            label_prediction.append(str(p))

    result = df.copy()

    result["Prediksi"] = label_prediction

    # ======================================================
    # PROBABILITY
    # ======================================================

    for i, cls in enumerate(model.classes_):

        if cls in LABEL_MAP:

            nama = LABEL_MAP[cls]

        else:

            nama = str(cls)

        result[f"Prob_{nama}"] = probability[:, i]

    st.session_state.prediction = result

    st.success("Prediksi berhasil dilakukan.")

    st.markdown("---")

    # ======================================================
    # DASHBOARD
    # ======================================================

    st.subheader("Ringkasan Prediksi")

    total = len(result)

    tinggi = (result["Prediksi"] == "Tinggi").sum()

    sedang = (result["Prediksi"] == "Sedang").sum()

    rendah = (result["Prediksi"] == "Rendah").sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total",
        total
    )

    c2.metric(
        "Tinggi",
        tinggi
    )

    c3.metric(
        "Sedang",
        sedang
    )

    c4.metric(
        "Rendah",
        rendah
    )

    st.markdown("---")

    # ======================================================
    # BAR CHART
    # ======================================================

    st.subheader("Distribusi Prediksi")

    chart = result["Prediksi"].value_counts()

    st.bar_chart(chart)

    # ======================================================
    # PIE CHART
    # ======================================================

    st.subheader("Persentase Prediksi")

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(

        chart.values,

        labels=chart.index,

        autopct="%1.1f%%",

        startangle=90

    )

    ax.axis("equal")

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================================
    # FILTER DATA
    # ======================================================

    st.subheader("Filter Hasil")

    keyword = st.text_input(
        "Cari Nama Siswa"
    )

    filtered = result.copy()

    if "nama_siswa" in filtered.columns:

        if keyword != "":

            filtered = filtered[
                filtered["nama_siswa"]
                .str.contains(
                    keyword,
                    case=False,
                    na=False
                )
            ]

    # ======================================================
    # TABEL HASIL
    # ======================================================

    tampil = []

    for col in [

        "student_id",

        "nama_siswa",

        "kelas",

        "semester",

        "Prediksi"

    ]:

        if col in filtered.columns:

            tampil.append(col)

    tampil.extend([

        "Prob_Rendah",

        "Prob_Sedang",

        "Prob_Tinggi"

    ])

    tampil = [

        c for c in tampil

        if c in filtered.columns

    ]

    st.subheader("Hasil Prediksi")

    st.dataframe(

        filtered[tampil],

        use_container_width=True,

        height=450

    )

    st.markdown("---")
# ==========================================================
# STEP 3
# FEATURE IMPORTANCE
# ==========================================================

st.header("📊 STEP 3 - Explainable AI")

coef = np.abs(model.coef_).mean(axis=0)

importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance": coef

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

st.subheader("Feature Importance")

st.dataframe(

    importance,

    use_container_width=True

)

st.bar_chart(

    importance.set_index("Feature")
)
# ==========================================================
# STEP 4
# EARLY WARNING SYSTEM
# ==========================================================

st.header("🚨 STEP 4 - Early Warning System")

warning = st.session_state.prediction.copy()

# ======================================================
# RISK LEVEL
# ======================================================

def risk_level(pred):

    if pred == "Rendah":

        return "🔴 High"

    elif pred == "Sedang":

        return "🟡 Medium"

    else:

        return "🟢 Low"

warning["Risk Level"] = warning["Prediksi"].apply(risk_level)

# ======================================================
# REKOMENDASI
# ======================================================

def recommendation(pred):

    if pred == "Rendah":

        return (
            "Segera lakukan pendampingan akademik, "
            "evaluasi aktivitas belajar, dan "
            "monitor perkembangan siswa."
        )

    elif pred == "Sedang":

        return (
            "Lakukan monitoring berkala serta "
            "berikan motivasi agar performa meningkat."
        )

    else:

        return (
            "Pertahankan performa belajar dan "
            "terus pantau perkembangan siswa."
        )

warning["Rekomendasi"] = warning["Prediksi"].apply(recommendation)

# ======================================================
# RINGKASAN
# ======================================================

high = (warning["Risk Level"]=="🔴 High").sum()

medium = (warning["Risk Level"]=="🟡 Medium").sum()

low = (warning["Risk Level"]=="🟢 Low").sum()

c1,c2,c3 = st.columns(3)

c1.metric(

    "🔴 High Risk",

    high

)

c2.metric(

    "🟡 Medium Risk",

    medium

)

c3.metric(

    "🟢 Low Risk",

    low

)

st.markdown("---")

# ======================================================
# FILTER
# ======================================================

option = st.selectbox(

    "Tampilkan",

    [

        "Semua",

        "🔴 High",

        "🟡 Medium",

        "🟢 Low"

    ]

)

if option != "Semua":

    warning = warning[
        warning["Risk Level"] == option
    ]

# ======================================================
# TABEL
# ======================================================

show = []

for col in [

    "student_id",

    "nama_siswa",

    "kelas",

    "semester",

    "Prediksi",

    "Risk Level",

    "Rekomendasi"

]:

    if col in warning.columns:

        show.append(col)

st.dataframe(

    warning[show],

    use_container_width=True,

    height=450

)
# ==========================================================
# STEP 5
# EXPORT REPORT
# ==========================================================

st.header("📄 STEP 5 - Export Report")

report = st.session_state.prediction.copy()

# Tambahkan Risk Level jika belum ada
if "Risk Level" not in report.columns:

    def risk_level(pred):

        if pred == "Rendah":
            return "🔴 High"

        elif pred == "Sedang":
            return "🟡 Medium"

        else:
            return "🟢 Low"

    report["Risk Level"] = report["Prediksi"].apply(risk_level)

# Tambahkan Rekomendasi jika belum ada
if "Rekomendasi" not in report.columns:

    def recommendation(pred):

        if pred == "Rendah":
            return "Pendampingan akademik segera."

        elif pred == "Sedang":
            return "Monitoring perkembangan belajar."

        else:
            return "Pertahankan performa belajar."

    report["Rekomendasi"] = report["Prediksi"].apply(recommendation)

st.subheader("Preview Report")

st.dataframe(
    report,
    use_container_width=True,
    height=350
)

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(

    label="⬇ Download Report (.csv)",

    data=csv,

    file_name="Early_Warning_Report.csv",

    mime="text/csv"

)

st.success("Report siap diunduh.")