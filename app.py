import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Credit Card Classification",
    layout="centered"
)

st.title("💳 Credit Card Default Classification")
st.write("Aplikasi Web untuk Training dan Testing Model Klasifikasi")

# =========================
# UPLOAD DATASET
# =========================
uploaded_file = st.file_uploader(
    "Upload Dataset (.xlsx atau .csv)",
    type=["xlsx", "csv"]
)

# =========================
# JIKA DATASET SUDAH DIUPLOAD
# =========================
if uploaded_file:

    # ---------- Load file ----------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset berhasil dimuat")

    # =========================
    # PREPROCESSING
    # =========================
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    X = df.drop(columns=["Y"])
    y = df["Y"]

    st.info("Preprocessing selesai")

    # =========================
    # PILIH RASIO DATA
    # =========================
    split_ratio = st.selectbox(
        "Pilih Rasio Data Train : Test",
        ["75 : 25", "80 : 20", "90 : 10"]
    )

    test_size = {
        "75 : 25": 0.25,
        "80 : 20": 0.20,
        "90 : 10": 0.10
    }[split_ratio]

    # =========================
    # TRAIN MODEL
    # =========================
    if st.button("🚀 Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        acc_train = accuracy_score(y_train, y_train_pred)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/classifier.pkl")
        joblib.dump(scaler, "model/scaler.pkl")

        st.success(f"Training selesai! Akurasi Training: {acc_train:.4f}")

    # =========================
    # TEST MODEL
    # =========================
    if st.button("🧪 Test Model"):

        model = joblib.load("model/classifier.pkl")
        scaler = joblib.load("model/scaler.pkl")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # =========================
        # EVALUASI
        # =========================
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        st.subheader("📊 Evaluasi Model")
        st.write(f"Accuracy  : **{acc:.4f}**")
        st.write(f"Precision : **{prec:.4f}**")
        st.write(f"Recall    : **{rec:.4f}**")

        # =========================
        # CONFUSION MATRIX
        # =========================
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # =========================
        # OUTPUT PREDIKSI (INI YANG DIMINTA DOSEN)
        # =========================
        result_df = X_test.copy()
        result_df["Y_Actual"] = y_test.values
        result_df["Y_Predicted"] = y_pred
        result_df["Status"] = (
            result_df["Y_Actual"] == result_df["Y_Predicted"]
        ).map({True: "Benar", False: "Salah"})

        st.subheader("📋 Hasil Prediksi Data Testing")
        st.dataframe(result_df.head(20))

        # =========================
        # DOWNLOAD HASIL
        # =========================
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Prediksi (.csv)",
            data=csv,
            file_name="hasil_prediksi_testing.csv",
            mime="text/csv"
        )
