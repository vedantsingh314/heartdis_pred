import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("Get a quick prediction of heart disease risk based on your health indicators.")

# Sidebar Input Form
st.sidebar.header("📝 Input Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.radio("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 250, 150)
exang = st.sidebar.radio("Exercise-Induced Angina (exang)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [1, 2, 3])

# Prepare input as DataFrame
input_df = pd.DataFrame([{
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

# Load models
with open('logreg_pipeline.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('knn_pipeline.pkl', 'rb') as f:
    model2 = pickle.load(f)

# Predict
pred1 = model1.predict_proba(input_df)[0][1]
pred2 = model2.predict_proba(input_df)[0][1]
final_pred = (pred1 + pred2) / 2

# Display prediction
st.subheader("🧠 Prediction Result")
st.metric(label="Heart Disease Risk", value=f"{final_pred:.2%}", delta="🚨 High" if final_pred >= 0.5 else "✅ Low")
st.progress(final_pred)

if final_pred >= 0.5:
    st.warning("⚠️ The patient is at **high risk** of heart disease. Recommend immediate medical consultation.")
else:
    st.success("✅ The patient is at **low risk** of heart disease.")
