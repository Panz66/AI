import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Credit Card Default Classification",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body { background-color: #f5f7fb; }
.section {
    background-color: white;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.05);
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.metric-value {
    font-size: 34px;
    font-weight: bold;
    color: #2563eb;
}
.metric-label {
    font-size: 14px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("👥 Nama Kelompok")
st.sidebar.markdown("""
- Ipan Wardana  
- Aldy Naufal Almarhum  
- Fajar Nurburging
""")

st.sidebar.title("⚙️ Kontrol Aplikasi")

split_ratio = st.sidebar.selectbox(
    "Rasio Data Train : Test",
    ["70 : 30", "75 : 25", "80 : 20", "85 : 15", "90 : 10"]
)

test_size = {
    "70 : 30": 0.30,
    "75 : 25": 0.25,
    "80 : 20": 0.20,
    "85 : 15": 0.15,
    "90 : 10": 0.10
}[split_ratio]

train_btn = st.sidebar.button("🚀 Train Model")
test_btn = st.sidebar.button("🧪 Test Model")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="section">
    <h1>💳 Credit Card Default Classification</h1>
    <p>
    Aplikasi klasifikasi <b>default pembayaran kartu kredit</b>
    menggunakan algoritma <b>Random Forest</b>.
    Dataset dimuat otomatis dari sistem.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATASET OTOMATIS
# =====================================================
DATA_PATH = "data/default_of_credit_card_clients.xlsx"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset tidak ditemukan di folder data/")
    st.stop()

df = pd.read_excel(DATA_PATH)

# =====================================================
# PREPROCESSING
# =====================================================
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

X = df.drop(columns=["Y"])
y = df["Y"]

# =====================================================
# INFO DATASET
# =====================================================
st.markdown("""
<div class="section">
    <h3>📂 Informasi Dataset</h3>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data", df.shape[0])
c2.metric("Jumlah Atribut", X.shape[1])
c3.metric("Jumlah Kelas", y.nunique())

# =====================================================
# TRAIN MODEL
# =====================================================
if train_btn:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    acc_train = accuracy_score(y_train, model.predict(X_train_scaled))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/classifier.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    st.success(f"✅ Training selesai | Akurasi Training: {acc_train:.4f}")

# =====================================================
# TEST MODEL
# =====================================================
if test_btn:
    if not os.path.exists("model/classifier.pkl"):
        st.warning("⚠️ Model belum ditraining")
        st.stop()

    model = joblib.load("model/classifier.pkl")
    scaler = joblib.load("model/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # ================= METRIC =================
    st.markdown("""
    <div class="section">
        <h3>📊 Evaluasi Model</h3>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, val, label in zip(
        [m1, m2, m3],
        [acc, prec, rec],
        ["Accuracy", "Precision", "Recall"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:.4f}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 9},
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ================= HASIL PREDIKSI =================
    result_df = X_test.copy()
    result_df["Y_Actual"] = y_test.values
    result_df["Y_Predicted"] = y_pred

    st.markdown("""
    <div class="section">
        <h3>📋 Hasil Prediksi Data Testing</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(result_df.head(30), use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Prediksi (.csv)",
        data=csv,
        file_name="hasil_prediksi_testing.csv",
        mime="text/csv"
    )
