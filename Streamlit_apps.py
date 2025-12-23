import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD MODEL
# =========================================================
st.title("Fatigue Prediction - Gradient Boosting Model Evaluation")
st.write("This Streamlit app loads a trained model and evaluates it using uploaded test data.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('optimized_gradient_boosting_model.pkl')
        config = joblib.load('gradient_boosting_best_config.pkl')
        params = joblib.load('gradient_boosting_best_params.pkl')
        return model, config, params
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, config, params = load_model()
if model is None:
    st.stop()

threshold = 0.5
if isinstance(config, dict):
    threshold = config.get("optimal_threshold", 0.5)

st.sidebar.header("Configuration")
st.sidebar.write(f"Optimal Threshold: **{threshold:.3f}**")

# =========================================================
# UPLOAD DATA
# =========================================================
st.header("Upload Test Dataset")
file = st.file_uploader("Upload CSV", type=['csv'])

if file:
    data_test = pd.read_csv(file)
    st.write("### Preview Data")
    st.dataframe(data_test.head())

    # FEATURES
    FEATURES_NUM = ["fatigue_count", "gap", "humidity"]
    FEATURES_CAT = ["conditions"]
    TARGET = "next_week_fatigue"    # OPTIONAL

    COST = {"TP": 1096, "FP": 550, "FN": 1820, "TN": 0}

    required_features = FEATURES_NUM + FEATURES_CAT
    missing = [c for c in required_features if c not in data_test.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # =========================================================
    # PREDIKSI TANPA ACTUAL
    # =========================================================
    X_test = data_test[required_features]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    st.write(f"### Total Samples: **{len(X_test)}**")

    # =========================================================
    # CONFUSION MATRIX JIKA KOLOM ACTUAL VALID
    # =========================================================
    if TARGET in data_test.columns:
        y_test_raw = data_test[TARGET]

        # Hanya ambil baris yang valid (0/1)
        valid_mask = y_test_raw.isin([0, 1])
        y_test_clean = y_test_raw[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_test_clean) > 0:
            st.subheader("Confusion Matrix (Actual Valid Only)")

            tn, fp, fn, tp = confusion_matrix(
                y_test_clean.astype(int),
                y_pred_clean.astype(int)
            ).ravel()

            cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                                 columns=["Pred 0", "Pred 1"],
                                 index=["Actual 0", "Actual 1"])
            st.dataframe(cm_df)

            # COST
            tp_cost = tp * COST["TP"]
            fp_cost = fp * COST["FP"]
            fn_cost = fn * COST["FN"]
            total_cost = tp_cost + fp_cost + fn_cost

    

    # =========================================================
    # VISUALISASI PREDIKSI
    # =========================================================
    st.header("Visualisasi Prediksi (Tanpa Actual)")

    # Summary
    st.subheader("Tabel Prediksi")
    pred_summary = pd.DataFrame({
        "Kategori": ["Prediksi 0 (Tidak Fatigue)", "Prediksi 1 (Fatigue)", "Total"],
        "Jumlah": [(y_pred == 0).sum(), (y_pred == 1).sum(), len(y_pred)]
    })
    st.dataframe(pred_summary)

    # Pie Chart
    st.subheader("Presentase Prediksi Fatigue vs Tidak Fatigue")
    values = [(y_pred == 0).sum(), (y_pred == 1).sum()]
    labels = [
        f"Tidak Fatigue (0)\n{values[0]} sampel",
        f"Fatigue (1)\n{values[1]} sampel"
    ]

    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Barplot CONDITIONS
    st.subheader("Distribusi CONDITIONS (Hanya Prediksi Fatigue)")
    fatigue_rows = data_test[y_pred == 1]
    cond_counts = fatigue_rows["conditions"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=cond_counts.index, y=cond_counts.values, ax=ax2)
    ax2.set_xlabel("Conditions")
    ax2.set_ylabel("Jumlah")
    ax2.set_title("Conditions untuk Prediksi Fatigue")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Histogram humidity
    st.subheader("Distribusi Humidity (Hanya Prediksi Fatigue)")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(fatigue_rows["humidity"], bins=10, ax=ax3, kde=True)
    ax3.set_xlabel("Humidity")
    ax3.set_ylabel("Frekuensi")
    ax3.set_title("Humidity pada Prediksi Fatigue")
    st.pyplot(fig3)

