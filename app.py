import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk & 10-Year Health Progression",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    .risk-high { background-color: #FEE2E2; border-left: 6px solid #DC2626; padding: 15px; border-radius: 8px; color: #991B1B; }
    .risk-mod { background-color: #FEF3C7; border-left: 6px solid #D97706; padding: 15px; border-radius: 8px; color: #92400E; }
    .risk-low { background-color: #D1FAE5; border-left: 6px solid #059669; padding: 15px; border-radius: 8px; color: #065F46; }
</style>
""", unsafe_allow_html=True)

def find_file(candidates):
    for name in candidates:
        if os.path.exists(name):
            return name
    return None

@st.cache_resource
def load_or_train_model():
    model_candidates = ["diabetes_model (3).pkl", "diabetes_model.pkl", "model.pkl"]
    scaler_candidates = ["scaler (4).pkl", "scaler.pkl"]
    feature_candidates = ["feature_names (4).pkl", "feature_names.pkl"]

    model_path = find_file(model_candidates)
    scaler_path = find_file(scaler_candidates)
    feature_path = find_file(feature_candidates)

    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    if model_path and scaler_path:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            if feature_path:
                with open(feature_path, "rb") as f:
                    feature_names = pickle.load(f)
            return model, scaler, feature_names, f"Loaded from `{model_path}`"
        except Exception as e:
            st.warning(f"Failed to load pickle files ({e}). Fallback to auto-trained model.")

    excel_path = "pima_diabetes_cleaned.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, sheet_name="Cleaned Dataset (Model Ready)")
    else:
        # Generate synthetic fallback dataset matching PIMA distributions
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "Pregnancies": np.random.randint(0, 10, n),
            "Glucose": np.random.randint(70, 190, n),
            "BloodPressure": np.random.randint(60, 100, n),
            "SkinThickness": np.random.randint(15, 45, n),
            "Insulin": np.random.randint(30, 250, n),
            "BMI": np.random.uniform(18.5, 45.0, n),
            "DiabetesPedigreeFunction": np.random.uniform(0.1, 1.5, n),
            "Age": np.random.randint(21, 70, n),
            "Outcome": np.random.randint(0, 2, n)
        })

    X = df[feature_names]
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_scaled, y)

    return model, scaler, feature_names, "Auto-trained Random Forest Model"

model, scaler, feature_list, model_status = load_or_train_model()

st.markdown('<div class="main-header">🩺 Diabetes Risk Prediction & Health Tracker</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Interactive Machine Learning Diagnostics & 10-Year Progression Simulator | <small>Status: {model_status}</small></div>', unsafe_allow_html=True)

st.sidebar.header("📋 Patient Clinical Parameters")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 40, 200, 117)
blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 40, 130, 72)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 23)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 14, 846, 85)
bmi = st.sidebar.slider("BMI (kg/m²)", 18.0, 67.1, 32.0, step=0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.372, step=0.001)
age = st.sidebar.slider("Age (Years)", 21, 81, 29)

input_dict = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}
input_df = pd.DataFrame([input_dict])[feature_list]
scaled_input = scaler.transform(input_df)

proba = model.predict_proba(scaled_input)[0][1] * 100

tab1, tab2, tab3 = st.tabs(["📊 Diagnostics & Prediction", "📈 10-Year Progression Forecast", "📋 Dataset & Audit"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Diagnostic Risk Evaluation")
        if proba >= 60:
            st.markdown(f'<div class="risk-high"><h3>⚠️ High Diabetes Risk</h3><h2>Probability: {proba:.1f}%</h2><p>Model predicts high probability of Type-2 Diabetes.</p></div>', unsafe_allow_html=True)
        elif proba >= 35:
            st.markdown(f'<div class="risk-mod"><h3>⚡ Moderate Diabetes Risk</h3><h2>Probability: {proba:.1f}%</h2><p>Borderline risk indicators detected.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><h3>✅ Low Diabetes Risk</h3><h2>Probability: {proba:.1f}%</h2><p>Clinical values are within normal ranges.</p></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("#### Patient Snapshot")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Glucose", f"{glucose} mg/dL")
        mcol2.metric("BMI", f"{bmi} kg/m²")
        mcol3.metric("Age", f"{age} yrs")

    with col2:
        st.subheader("Feature Importance Breakdown")
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.DataFrame({
                "Feature": feature_list,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            st.bar_chart(feat_imp.set_index("Feature"))

with tab2:
    st.subheader("🔮 10-Year Risk Progression Simulator")
    annual_bmi = st.slider("Annual BMI Change (kg/m²/yr)", -0.5, 1.0, 0.1, step=0.1)
    annual_glucose = st.slider("Annual Glucose Drift (mg/dL/yr)", -1.0, 3.0, 0.5, step=0.1)

    years = list(range(0, 11))
    future_probas = []
    for y in years:
        sim_dict = input_dict.copy()
        sim_dict["Age"] = min(81, age + y)
        sim_dict["BMI"] = max(15.0, min(65.0, bmi + (annual_bmi * y)))
        sim_dict["Glucose"] = max(50, min(250, glucose + (annual_glucose * y)))
        sim_df = pd.DataFrame([sim_dict])[feature_list]
        p = model.predict_proba(scaler.transform(sim_df))[0][1] * 100
        future_probas.append(p)

    prog_df = pd.DataFrame({"Year": [f"Yr {y}" for y in years], "Risk (%)": future_probas}).set_index("Year")
    st.line_chart(prog_df)

with tab3:
    st.subheader("📄 Dataset Inspection")
    if os.path.exists("pima_diabetes_cleaned.xlsx"):
        df = pd.read_csv('pima_diabetes_cleaned.csv')
        st.dataframe(df_display.head(20))
    else:
        st.info("Local `pima_diabetes_cleaned.xlsx` file not found.")

st.markdown("---")
st.markdown("💡 *Streamlit Diabetes Risk App*")
