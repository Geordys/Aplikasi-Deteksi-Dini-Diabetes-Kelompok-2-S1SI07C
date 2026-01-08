# =========================================================
# APLIKASI DETEKSI DINI DIABETES
# Tugas Besar Data Mining - Kelompok 2 S1SI-07-C
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Deteksi Dini Diabetes", layout="wide")
st.title("🩺 Aplikasi Deteksi Dini Diabetes")

st.markdown("""
Aplikasi ini dibuat untuk memenuhi **Tugas Besar Mata Kuliah Data Mining**  
menggunakan metode **Logistic Regression**.
""")

# =========================================================
# DATA KELOMPOK
# =========================================================
st.subheader("👥 Data Kelompok")
st.markdown("""
- **Kelompok** : Kelompok 2  
- **Kelas** : S1SI-07-C  
- **Mata Kuliah** : Data Mining  
- **Topik** : Deteksi Dini Diabetes  
""")

# =========================================================
# LOAD DATASET
# =========================================================
st.subheader("📂 Load Dataset")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes_data_upload.csv")

data = load_data()
st.dataframe(data)

# =========================================================
# DATA EXPLORATION
# =========================================================
st.subheader("🔍 Data Exploration")

st.write("Jumlah Data:", data.shape[0])
st.write("Jumlah Atribut:", data.shape[1])

# Missing value check
if data.isnull().sum().sum() == 0:
    st.success("✅ Tidak terdapat missing value pada dataset")
else:
    st.warning("⚠️ Terdapat missing value")

# =========================================================
# DATA CLEANING & ENCODING
# =========================================================
st.subheader("🧹 Data Cleaning & Encoding")

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

st.success("✅ Encoding data kategorikal selesai")

# =========================================================
# OUTLIER HANDLING (AGE)
# =========================================================
st.subheader("📊 Handling Outlier (Age - IQR)")

Q1 = data["Age"].quantile(0.25)
Q3 = data["Age"].quantile(0.75)
IQR = Q3 - Q1

data = data[
    (data["Age"] >= (Q1 - 1.5 * IQR)) &
    (data["Age"] <= (Q3 + 1.5 * IQR))
]

st.success("✅ Outlier pada kolom Age berhasil ditangani")

# =========================================================
# DATA PREPARATION
# =========================================================
st.subheader("⚙️ Data Preparation")

X = data.drop("class", axis=1)
y = data["class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

st.success("✅ Data berhasil dipisahkan (Train & Test)")

# =========================================================
# MODELLING
# =========================================================
st.subheader("🤖 Modelling (Logistic Regression)")

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.success("✅ Model Logistic Regression berhasil dilatih")

# =========================================================
# EVALUATION
# =========================================================
st.subheader("📈 Evaluation Model")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"**Accuracy** : {acc:.2f}")
st.write(f"**Precision** : {prec:.2f}")
st.write(f"**Recall** : {rec:.2f}")
st.write(f"**F1-Score** : {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# =========================================================
# PENUTUP
# =========================================================
st.markdown("---")
st.markdown("📌 *Aplikasi ini dikembangkan menggunakan Streamlit dan di-deploy melalui Streamlit Cloud.*")
