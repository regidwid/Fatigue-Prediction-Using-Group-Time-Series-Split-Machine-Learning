import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# LOAD MODEL
# =========================================================
st.title("Fatigue Prediction - Gradient Boosting Model Evaluation")
st.write("This Streamlit app loads a trained model and evaluates it using uploaded test data.")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('optimized_gradient_boosting_model.pkl')
        config = joblib.load('gradient_boosting_best_config.pkl')
        params = joblib.load('gradient_boosting_best_params.pkl')
        return model, config, params
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, config, params = load_model()

if model is None:
    st.stop()

# Extract threshold
if isinstance(config, dict):
    threshold = config.get('optimal_threshold', 0.5)
else:
    threshold = 0.5

st.sidebar.header("Configuration")
st.sidebar.write(f"Optimal Threshold: **{threshold:.3f}**")

# =========================================================
# UPLOAD TEST DATA
# =========================================================
st.header("Upload Test Dataset")
file = st.file_uploader("Upload CSV file", type=['csv'])

if file:
    data_test = pd.read_csv(file)
    st.write("### Preview Data")
    st.dataframe(data_test.head())

    FEATURES_NUM = ["fatigue_count", "gap", "humidity"]
    FEATURES_CAT = ["conditions"]
    TARGET = "next_week_fatigue"

    COST = {"TP": 1096, "FP": 550, "FN": 1820, "TN": 0}

    # Check required columns
    required = FEATURES_NUM + FEATURES_CAT + [TARGET]
    missing = [c for c in required if c not in data_test.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Clean
    data_test_clean = data_test.dropna(subset=[TARGET]).copy()

    X_test = data_test_clean[FEATURES_NUM + FEATURES_CAT]
    y_test = data_test_clean[TARGET].astype(int)

    st.write(f"Total Samples: {len(X_test)}")

    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                         columns=["Pred 0", "Pred 1"],
                         index=["Actual 0", "Actual 1"])
    st.dataframe(cm_df)

    # Cost analysis
    tp_cost = tp * COST["TP"]
    fp_cost = fp * COST["FP"]
    fn_cost = fn * COST["FN"]
    total_cost = tp_cost + fp_cost + fn_cost
    cost_sample = total_cost / len(y_test)

    st.subheader("Cost Analysis")
    st.write(f"TP Cost : **{tp_cost}**")
    st.write(f"FP Cost: **{fp_cost}**")
    st.write(f"FN Cost: **{fn_cost}**")
    st.write(f"### Total Cost: **{total_cost}**")
    st.write(f"Cost per Sample: **{cost_sample:.2f}**")

# =========================================================
# DASHBOARD VISUALIZATION (NO ACTUAL LABEL AVAILABLE)
# =========================================================

st.header("📊 Fatigue Prediction Dashboard")

result_df = X_test.copy()
result_df["pred_proba"] = y_proba
result_df["predicted"] = y_pred

# -------------------------
# 1. Pie Chart: Distribusi Predicted Fatigue
# -------------------------
st.subheader("1️⃣ Persentase Prediksi Fatigue")

fig1, ax1 = plt.subplots(figsize=(5, 5))
result_df["predicted"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Tidak Fatigue (0)", "Fatigue (1)"],
    ax=ax1
)
ax1.set_ylabel("")
ax1.set_title("Distribusi Prediksi Fatigue")
st.pyplot(fig1)

# -------------------------
# 2. Histogram: Distribusi Probabilitas Prediksi
# -------------------------
st.subheader("2️⃣ Distribusi Probabilitas Prediksi Fatigue")

fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(result_df["pred_proba"], kde=True, ax=ax2)
ax2.set_title("Distribusi Probabilitas Prediksi")
ax2.set_xlabel("Predicted Probability")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# -------------------------
# 3. Rata-rata probabilitas per kondisi cuaca
# -------------------------
st.subheader("3️⃣ Rata-rata Probabilitas Fatigue per Kondisi Cuaca")

weather_avg = result_df.groupby("conditions")["pred_proba"].mean().sort_values()

fig3, ax3 = plt.subplots(figsize=(7, 4))
weather_avg.plot(kind="bar", ax=ax3)
ax3.set_title("Rata-rata Probabilitas Fatigue Berdasarkan Cuaca")
ax3.set_xlabel("Kondisi Cuaca")
ax3.set_ylabel("Probabilitas Rata-rata")
st.pyplot(fig3)

# -------------------------
# 4. Bar Chart: Jumlah Prediksi Fatigue per Kondisi Cuaca
# -------------------------
st.subheader("4️⃣ Distribusi Prediksi Fatigue per Kondisi Cuaca")

fatigue_by_weather = result_df.groupby("conditions")["predicted"].sum().sort_values()

fig4, ax4 = plt.subplots(figsize=(7, 4))
fatigue_by_weather.plot(kind="bar", ax=ax4)
ax4.set_title("Jumlah Prediksi Fatigue per Kondisi Cuaca")
ax4.set_xlabel("Kondisi Cuaca")
ax4.set_ylabel("Jumlah Prediksi Fatigue")
st.pyplot(fig4)

# -------------------------
# 5. Boxplot: Distribusi fitur numerik
# -------------------------
st.subheader("5️⃣ Distribusi Fitur Numerik")

for col in FEATURES_NUM:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=result_df, y=col, ax=ax)
    ax.set_title(f"Distribusi Fitur: {col}")
    ax.set_ylabel(col)
    st.pyplot(fig)

# -------------------------
# 6. Scatter: Hubungan fitur numerik vs probabilitas
# -------------------------
st.subheader("6️⃣ Hubungan Fitur vs Probabilitas Prediksi")

for col in FEATURES_NUM:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=result_df, x=col, y="pred_proba", ax=ax)
    ax.set_title(f"{col} vs Probabilitas Fatigue")
    ax.set_xlabel(col)
    ax.set_ylabel("Pred Probability")
    st.pyplot(fig)

# -------------------------
# 7. Heatmap korelasi fitur + prediksi
# -------------------------
st.subheader("7️⃣ Korelasi Fitur dengan Probabilitas Prediksi")

corr = result_df[FEATURES_NUM + ["pred_proba"]].corr()

fig7, ax7 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax7)
ax7.set_title("Heatmap Korelasi")
st.pyplot(fig7)

st.success("🎉 Dashboard visualisasi prediksi fatigue (tanpa actual) berhasil ditambahkan!")
