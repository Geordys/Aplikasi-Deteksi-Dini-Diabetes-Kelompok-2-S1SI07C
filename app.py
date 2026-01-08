import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import statsmodels.api as sm

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================
st.set_page_config(page_title="Diabetes AI System", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'lang' not in st.session_state:
    st.session_state.lang = 'ID'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

def toggle_lang():
    st.session_state.lang = 'EN' if st.session_state.lang == 'ID' else 'ID'

en = st.session_state.lang == 'EN'

# ==========================================
# 2. LOAD DATA & TRAIN MODEL
# ==========================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("diabetes_data_upload.csv")
    df = df.dropna()
    df_raw = df.copy()

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop('class', axis=1)
    y = df['class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return df_raw, df, model, X_test, y_test, scaler, X.columns

df_raw, df_encoded, logreg, X_test, y_test, scaler, feature_names = load_and_train()
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# ==========================================
# 3. DASHBOARD UTAMA (ISI TIDAK DIUBAH)
# ==========================================
st.title("🏥 Sistem Deteksi Dini Diabetes AI")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Data", len(df_raw))
k2.metric("Kasus Positif", f"{(df_raw['class']=='Positive').mean():.1%}")
k3.metric("Rerata Usia", f"{df_raw['Age'].mean():.1f}")
k4.metric("Akurasi AI", f"{accuracy_score(y_test, y_pred):.1%}")

v1, v2 = st.columns(2)

with v1:
    st.subheader("🚻 Komposisi Gender")
    st.plotly_chart(px.pie(df_raw, names="Gender", hole=0.5), use_container_width=True)

with v2:
    st.subheader("📈 Distribusi Usia & Diagnosa")
    st.plotly_chart(
        px.histogram(df_raw, x="Age", color="class", barmode="group"),
        use_container_width=True
    )

# ==========================================
# 4. FORM PREDIKSI (ISI TIDAK DIUBAH)
# ==========================================
st.subheader("🩺 AI Prediktor Risiko Diabetes")

with st.form("predict_form"):
    age = st.slider("Usia", 10, 90, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    obesity = st.selectbox("Obesity", ["No", "Yes"])

    submit = st.form_submit_button("CEK HASIL")

if submit:
    bm = lambda x: 1 if x in ["Yes", "Male"] else 0
    input_data = [age, bm(gender)] + [0]*13 + [bm(obesity)]
    prob = logreg.predict_proba(scaler.transform([input_data]))[0][1]

    st.success(f"Probabilitas Risiko Diabetes: **{prob:.1%}**")
