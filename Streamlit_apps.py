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
    st.write(f"TP Cost (Gain): **{tp_cost}**")
    st.write(f"FP Cost: **{fp_cost}**")
    st.write(f"FN Cost: **{fn_cost}**")
    st.write(f"### Total Cost: **{total_cost}**")
    st.write(f"Cost per Sample: **{cost_sample:.2f}**")

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()

    st.pyplot(fig)
