# Fatigue Prediction Streamlit App

This repository contains a **Streamlit web application** for evaluating a trained **Gradient Boosting fatigue prediction model**. The app loads a stored model, processes uploaded test data, computes predictions, calculates cost-based evaluation, and visualizes performance.

---

## 🚀 Features

* Upload CSV test dataset
* Automatic data validation
* Probability prediction using Gradient Boosting model
* Apply optimal threshold from saved config
* Confusion matrix visualization
* ROC curve plot
* Full cost-based evaluation using custom cost matrix

---

## 📦 Project Structure

```
📁 project
│── streamlit_app.py
│── README.md
│── optimized_gradient_boosting_model.pkl
│── gradient_boosting_best_config.pkl
│── gradient_boosting_best_params.pkl
```

---

## ▶️ Running the Streamlit App

Make sure Python 3.9+ is installed.

### 1. Install dependencies

```
pip install -r requirements.txt
```

Typical requirements:

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

### 2. Run the app

```
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (default: `http://localhost:8501`).

---

## 📤 Uploading Test Data

Your CSV must contain these columns:

* `fatigue_count`
* `gap`
* `humidity`
* `conditions`
* `next_week_fatigue`

Missing columns will stop execution.

---

## ⚙️ Cost Matrix Used

The evaluation uses the following cost structure:

```
TP = 1096   (gain)
FP = 550    (cost)
FN = 1820   (cost)
TN = 0
```

The app computes:

* Total cost
* Cost per sample
* Contribution from each component (TP, FP, FN)

---

## 📊 Visualizations

The app provides:

* Confusion Matrix
* ROC Curve (with AUC)

All plots are embedded directly in the Streamlit interface.

---

## ✨ Notes

* Ensure the `.pkl` model files are placed in the same directory as the Streamlit app.
* This repository is ready for deployment to **Streamlit Cloud** or **GitHub Pages (via Streamlit deployment)**.

---

## 📞 Support

If you need help integrating this with a full machine learning pipeline or preparing deployment, feel free to ask!

## 🌟 GitHub Repository Description

A Streamlit-based web application for evaluating a Gradient Boosting Machine (GBM) fatigue prediction model used in industrial and mining operational safety. This app allows users to upload test datasets, run predictions using a pre-trained model, perform cost-based evaluation, and visualize key metrics including confusion matrix and ROC curve. Ideal for machine learning deployment, safety analysis, and fatigue risk management systems.
