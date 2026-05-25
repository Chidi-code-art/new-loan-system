import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainable AI Loan Approval System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD CUSTOM CSS
# =========================================================

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL + TRAINING DATA
# =========================================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

x_train = pd.read_csv("x_train.csv")

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">
    <h1> Explainable AI Loan Approval System</h1>
    <p>
        Predict loan approval using Machine Learning with
        SHAP and LIME Explainability
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# SIDEBAR - USER FRIENDLY INPUTS
# =========================================================
st.sidebar.markdown("##  Applicant Information")

age = st.sidebar.slider("Age", 18, 80, 30)
income = st.sidebar.number_input("Monthly Income ($)", min_value=1000, max_value=1000000, value=50000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

# === Gender ===
gender_map = {"Female": 0, "Male": 1}
new_gender_label = st.sidebar.selectbox("Gender", options=list(gender_map.keys()))
new_gender = gender_map[new_gender_label]

# === Marital Status ===
marital_map = {"Single": 0, "Married": 1}
new_marital_label = st.sidebar.selectbox("Marital Status", options=list(marital_map.keys()))
new_marital_status = marital_map[new_marital_label]

# === Education Level ===
education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's or Higher": 2
}
new_education_label = st.sidebar.selectbox("Education Level", options=list(education_map.keys()))
new_education_level = education_map[new_education_label]

# === Occupation ===
occupation_map = {
    "Unemployed": 0,
    "Student": 1,
    "Entry Level": 2,
    "Professional": 3
    # Add more categories as per your actual encoding
}
new_occupation_label = st.sidebar.selectbox("Occupation", options=list(occupation_map.keys()))
new_occupation = occupation_map[new_occupation_label]

# =========================================================
# INPUT DATAFRAME
# =========================================================
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'credit_score': [credit_score],
    'new_gender': [new_gender],
    'new_occupation': [new_occupation],
    'new_education_level': [new_education_level],
    'new_marital_status': [new_marital_status]
})

# Ensure column order matches training data
input_data = input_data[x_train.columns]

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(" Predict Loan Approval"):

    prediction = model.predict(input_data)[0]

    prediction_proba = model.predict_proba(input_data)[0]

    approval_probability = prediction_proba[1]

    st.markdown("---")

    # =====================================================
    # RESULT CARDS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader(" Prediction Result")

        if prediction == 1:
            st.success(" Loan Approved")
        else:
            st.error(" Loan Denied")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader(" Approval Probability")

        st.metric(
            "Confidence Score",
            f"{approval_probability*100:.2f}%"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    st.markdown("##  SHAP Explainability")

    background_data = shap.sample(x_train, 50)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background_data
    )

    shap_values = explainer.shap_values(input_data)

    # Handle SHAP versions
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    # =====================================================
    # SHAP BAR PLOT
    # =====================================================

    st.markdown("### Global Feature Importance")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    shap.summary_plot(
        shap_values_class1,
        input_data,
        feature_names=input_data.columns.tolist(),
        plot_type="bar",
        show=False
    )

    st.pyplot(fig1)

    # =====================================================
    # SHAP WATERFALL (FIXED)
    # =====================================================

    st.markdown("### Local Prediction Explanation")

    # Extract scalar expected value for class 1
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        expected_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        expected_value = float(ev)

    # Extract 1D SHAP values for the single input instance (class 1)
    if isinstance(shap_values, list):
        shap_single = np.array(shap_values[1][0]).flatten()
    else:
        if shap_values.ndim == 3:
            shap_single = shap_values[0, :, 1].flatten()
        else:
            shap_single = shap_values[0].flatten()

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_single,
        feature_names=input_data.columns.tolist()
    )

    st.pyplot(fig2)

       # ===============================
    # LIME FIXED VERSION
    # ===============================
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=x_train.columns.tolist(),
        class_names=['Denied', 'Approved'],
        mode='classification'
    )
    
    # Convert input row to numpy array (VERY IMPORTANT FIX)
    input_row = input_data.values[0]
    
    exp = lime_explainer.explain_instance(
        data_row=input_row,
        predict_fn=model.predict_proba,
        num_features=7
    )
    
    lime_df = pd.DataFrame(
        exp.as_list(),
        columns=['Feature', 'Contribution']
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.markdown("##  Model Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by='Importance',
            ascending=False
        )

        fig4 = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance'
        )

        st.plotly_chart(fig4, use_container_width=True)

    else:

        st.info(
            "Logistic Regression does not provide feature_importances_."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <h4>Explainable AI Loan Approval System</h4>
    <p>Built with Streamlit • SHAP • LIME • Machine Learning</p>
</div>
""", unsafe_allow_html=True)
