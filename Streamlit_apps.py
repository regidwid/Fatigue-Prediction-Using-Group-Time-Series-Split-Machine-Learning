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
# DASHBOARD VISUALIZATION
# =========================================================

st.header("📊 Fatigue Dashboard Visualization")

# Gabungkan hasil prediksi dengan data asli
result_df = data_test_clean.copy()
result_df["pred_proba"] = y_proba
result_df["predicted"] = y_pred

# -------------------------
# 1. Bar Chart: Jumlah Fatigue Actual vs Predicted
# -------------------------
st.subheader("1️⃣ Jumlah Fatigue: Actual vs Predicted")

fatigue_counts = pd.DataFrame({
    "Actual": result_df["next_week_fatigue"].value_counts(),
    "Predicted": result_df["predicted"].value_counts()
})

fig1, ax1 = plt.subplots(figsize=(6, 4))
fatigue_counts.plot(kind="bar", ax=ax1)
ax1.set_title("Actual vs Predicted Fatigue")
ax1.set_xlabel("Label")
ax1.set_ylabel("Jumlah")
st.pyplot(fig1)


# -------------------------
# 2. Pie Chart: Presentase Fatigue Actual
# -------------------------
st.subheader("2️⃣ Presentase Fatigue (Actual)")

fig2, ax2 = plt.subplots(figsize=(5, 5))
result_df["next_week_fatigue"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", labels=["Tidak Fatigue (0)", "Fatigue (1)"], ax=ax2
)
ax2.set_ylabel("")
ax2.set_title("Presentase Fatigue (Actual)")
st.pyplot(fig2)


# -------------------------
# 3. Pie Chart: Presentase Predicted Fatigue
# -------------------------
st.subheader("3️⃣ Presentase Prediksi Fatigue")

fig3, ax3 = plt.subplots(figsize=(5, 5))
result_df["predicted"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", labels=["Pred 0", "Pred 1"], ax=ax3
)
ax3.set_ylabel("")
ax3.set_title("Presentase Predicted Fatigue")
st.pyplot(fig3)


# -------------------------
# 4. Bar Chart: Rata-rata Probabilitas Fatigue per Kondisi Cuaca
# -------------------------
st.subheader("4️⃣ Rata-rata Probabilitas Fatigue per Kondisi Cuaca")

group_avg = result_df.groupby("conditions")["pred_proba"].mean().sort_values()

fig4, ax4 = plt.subplots(figsize=(7, 4))
group_avg.plot(kind="bar", ax=ax4)
ax4.set_title("Rata-rata Probabilitas Fatigue per Kondisi Cuaca")
ax4.set_xlabel("Cuaca")
ax4.set_ylabel("Probabilitas Fatigue")
st.pyplot(fig4)


# -------------------------
# 5. Histogram: Distribusi Prediksi Probabilitas
# -------------------------
st.subheader("5️⃣ Distribusi Probabilitas Prediksi")

fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.histplot(result_df["pred_proba"], kde=True, ax=ax5)
ax5.set_title("Distribusi Probability Fatigue")
ax5.set_xlabel("Predicted Probability")
ax5.set_ylabel("Frequency")
st.pyplot(fig5)


# -------------------------
# 6. Boxplot: Hubungan Fitur Numerik terhadap Fatigue
# -------------------------
st.subheader("6️⃣ Analisis Fitur Numerik berdasarkan Actual Fatigue")

for col in FEATURES_NUM:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=result_df, x="next_week_fatigue", y=col, ax=ax)
    ax.set_title(f"{col} vs Actual Fatigue")
    ax.set_xlabel("Actual Fatigue (0/1)")
    ax.set_ylabel(col)
    st.pyplot(fig)


# -------------------------
# 7. Breakdown TP / FP / FN / TN
# -------------------------
st.subheader("7️⃣ Breakdown TP / FP / FN / TN")

def case_type(row):
    if row["next_week_fatigue"] == 1 and row["predicted"] == 1:
        return "TP"
    if row["next_week_fatigue"] == 0 and row["predicted"] == 1:
        return "FP"
    if row["next_week_fatigue"] == 1 and row["predicted"] == 0:
        return "FN"
    return "TN"

result_df["case_type"] = result_df.apply(case_type, axis=1)

fig7, ax7 = plt.subplots(figsize=(6, 4))
sns.countplot(data=result_df, x="case_type", ax=ax7)
ax7.set_title("Error Case Breakdown (TP / FP / FN / TN)")
ax7.set_xlabel("Case Type")
ax7.set_ylabel("Count")
st.pyplot(fig7)


# -------------------------
# 8. Heatmap Korelasi Fitur
# -------------------------
st.subheader("8️⃣ Korelasi Fitur terhadap Probabilitas Fatigue")

corr = result_df[FEATURES_NUM + ["pred_proba"]].corr()

fig8, ax8 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax8)
ax8.set_title("Correlation Heatmap")
st.pyplot(fig8)

st.success("🎉 Dashboard visualisasi fatigue berhasil ditambahkan!")

