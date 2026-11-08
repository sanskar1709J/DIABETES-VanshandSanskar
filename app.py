import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Glucoview 10-Year Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- CACHED MODEL TRAINING PIPELINE ---
@st.cache_resource
def train_diabetes_models():
    """
    Generates realistic clinical training data and trains:
    1. An XGBoost Classifier wrapped in Isotonic Calibration (for 0-100% current risk).
    2. A Cox Proportional Hazards model (for the 10-year survival/risk timeline).
    """
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic clinical features
    age = np.random.randint(25, 75, size=n_samples)
    bmi = np.random.normal(27, 5, size=n_samples).clip(15, 50)
    glucose = np.random.normal(100, 25, size=n_samples).clip(70, 250)
    blood_pressure = np.random.normal(120, 15, size=n_samples).clip(90, 190)
    family_history = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    physical_activity = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2]) # 0: Low, 1: Mod, 2: High

    # Synthetic target generation logic based on risk factors
    risk_score = (
        (age - 25) * 0.04 +
        (bmi - 22) * 0.07 +
        (glucose - 90) * 0.06 +
        (blood_pressure - 110) * 0.02 +
        family_history * 1.2 -
        physical_activity * 0.5
    )
    # Convert to probability via sigmoid
    prob = 1 / (1 + np.exp(-risk_score + 3.5))
    current_diabetes = (np.random.random(n_samples) < prob).astype(int)
    
    # Time-to-event for survival analysis (in years, max 10)
    time_to_event = np.random.exponential(scale=8, size=n_samples)
    time_to_event = np.clip(time_to_event, 0.5, 10.0)
    # If they already have diabetes or low risk, adjust event times
    event_occurred = np.where((current_diabetes == 1) | (prob > 0.6), 1, np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]))
    
    df = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'FamilyHistory': family_history,
        'PhysicalActivity': physical_activity,
        'Target': current_diabetes,
        'Time': time_to_event,
        'Event': event_occurred
    })

    # Features for classification
    X_class = df[['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory', 'PhysicalActivity']]
    y_class = df['Target']
    
    scaler = StandardScaler()
    X_class_scaled = scaler.fit_transform(X_class)
    
    # Train XGBoost with CalibratedClassifierCV to get true 0-100% probabilities
    base_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_xgb, method='isotonic', cv=3)
    calibrated_model.fit(X_class_scaled, y_class)
    
    # Train Cox Proportional Hazards Model for Timeline Analysis
    X_surv = df[['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory', 'PhysicalActivity']]
    # scikit-survival requires structured array for (event, time)
    y_surv = np.array(list(zip(df['Event'].astype(bool), df['Time'])), dtype=[('Status', '?'), ('Time', '<f8')])
    
    cox_model = CoxPHSurvivalAnalysis()
    cox_model.fit(X_surv, y_surv)

    return calibrated_model, cox_model, scaler

# Load models
clf_model, cox_model, scaler = train_diabetes_models()

# --- STREAMLIT USER INTERFACE ---
st.title("🩺 Advanced Diabetes Risk & 10-Year Timeline Predictor")
st.markdown("""
This clinical decision-support tool utilizes an **XGBoost Classifier with Isotonic Probability Calibration** to eliminate clustering artifacts and provide true 0–100% risk scoring, alongside a **Cox Proportional Hazards Survival Model** to map cumulative risk progression over a 10-year timeline.
""")

st.sidebar.header("Patient Clinical Parameters")

def user_input_features():
    age = st.sidebar.slider("Age (Years)", 20, 80, 42)
    bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 45.0, 26.5)
    glucose = st.sidebar.slider("Fasting Blood Glucose (mg/dL)", 70, 250, 105)
    blood_pressure = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 90, 190, 122)
    family_history = st.sidebar.selectbox("Family History of Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    physical_activity = st.sidebar.selectbox("Physical Activity Level", options=[0, 1, 2], format_func=lambda x: ["Low (Sedentary)", "Moderate", "High (Active)"][x])
    
    data = {
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'FamilyHistory': family_history,
        'PhysicalActivity': physical_activity
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Input Summary
st.subheader("📋 Current Patient Profile")
st.dataframe(input_df, use_container_width=True)

# --- PREDICTION COMPUTATION ---
# Scale inputs for classification model
input_scaled = scaler.transform(input_df)

# Current probability (0 to 100%)
current_prob = clf_model.predict_proba(input_scaled)[0][1] * 100

# Layout metrics
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Estimated Current Diabetes Probability", value=f"{current_prob:.1f}%")
with col2:
    risk_category = "High Risk" if current_prob > 50 else ("Moderate Risk" if current_prob > 25 else "Low Risk")
    st.metric(label="Clinical Risk Stratification", value=risk_category)

st.markdown("---")

# --- 10-YEAR TIMELINE GRAPH GENERATION ---
st.subheader("📈 10-Year Cumulative Risk Timeline")
st.markdown("The curve below forecasts the cumulative percentage chance of developing or progressing with Type 2 Diabetes over a 10-year horizon based on your physiological parameters.")

# Predict survival function using Cox Model
try:
    survival_funcs = cox_model.predict_survival_function(input_df)
    timeline_years = np.linspace(0, 10, 50)
    
    # Calculate cumulative risk: 1 - Survival Probability
    cumulative_risk = [1 - fn(t) for t in timeline_years]
    # Scale based on current model output to blend baseline hazard smoothly
    cumulative_risk = np.clip(np.array(cumulative_risk) * (current_prob / 50.0), 0, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timeline_years, cumulative_risk, color='#ff4b4b', linewidth=3, label="Cumulative Risk Trajectory")
    ax.fill_between(timeline_years, 0, cumulative_risk, color='#ff4b4b', alpha=0.15)
    
    ax.set_title("10-Year Projected Diabetes Risk Progression", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Timeline (Years from baseline)", fontsize=10)
    ax.set_ylabel("Cumulative Risk (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)
except Exception as e:
    st.error(f"Could not render survival curve: {e}")

# Footer notes
st.markdown("---")
*st.caption("Disclaimer: This tool is built for educational and demonstration purposes using synthetic clinical data distributions. It does not replace professional medical diagnosis or endocrinological evaluation.")*
