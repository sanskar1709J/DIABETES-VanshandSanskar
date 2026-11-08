import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

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

@st.cache_resource
def load_or_train_model():
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    excel_path = "combined_diabetes_cleaned.xlsx"
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            df = df.dropna(subset=['Outcome', 'Glucose', 'BMI'])
        except Exception as e:
            st.warning(f"Error reading Excel file: {e}. Falling back to default URL...")
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            cols = feature_names + ["Outcome"]
            df = pd.read_csv(url, names=cols)
    else:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = feature_names + ["Outcome"]
        df = pd.read_csv(url, names=cols)

    X = df[feature_names]
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    model = CalibratedClassifierCV(base_rf, cv=3)
    model.fit(X_scaled, y)

    return model, scaler, feature_names, "Trained on combined_diabetes_cleaned.xlsx"

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

# Reduced tabs to 2 (removed Dataset & Audit)
tab1, tab2 = st.tabs(["📊 Diagnostics & Prediction", "📈 10-Year Progression Forecast"])

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
        estimator = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
        if hasattr(estimator, "feature_importances_"):
            feat_imp = pd.DataFrame({
                "Feature": feature_list,
                "Importance": estimator.feature_importances_
            }).sort_values("Importance", ascending=True)
            st.bar_chart(feat_imp.set_index("Feature"))

with tab2:
    st.subheader("🔮 10-Year Risk Progression Simulator")
    st.markdown("Adjust the expected annual rate of change to simulate future health trajectories.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        annual_bmi = st.slider("Annual BMI Change (kg/m²/yr)", -1.0, 2.0, 0.3, step=0.1)
    with col_s2:
        annual_glucose = st.slider("Annual Glucose Drift (mg/dL/yr)", -2.0, 6.0, 1.5, step=0.1)

    years = list(range(0, 11))
    future_probas = []
    
    for y in years:
        sim_dict = input_dict.copy()
        sim_dict["Age"] = int(min(81, age + y))
        sim_dict["BMI"] = float(max(15.0, min(67.1, bmi + (annual_bmi * y))))
        sim_dict["Glucose"] = float(max(40, min(250, glucose + (annual_glucose * y))))
        
        sim_df = pd.DataFrame([sim_dict])[feature_list]
        scaled_sim = scaler.transform(sim_df)
        
        p = model.predict_proba(scaled_sim)[0][1] * 100
        future_probas.append(round(p, 2))

    # Using zero-padded labels ("Year 00" to "Year 10") to guarantee correct chronological axis sorting
    prog_df = pd.DataFrame({
        "Year": [f"Year {y:02d}" for y in years],
        "Projected Risk (%)": future_probas
    })
    
    st.line_chart(prog_df.set_index("Year"), use_container_width=True)
    
    with st.expander("View Year-by-Year Projection Data"):
        # Display with cleaner labels in the table
        table_df = prog_df.copy()
        table_df["Year"] = [f"Year {y}" for y in years]
        st.dataframe(table_df, use_container_width=True)

st.markdown("---")
st.markdown("💡 *Streamlit Diabetes Risk App*")
