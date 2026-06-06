import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Smart Loan Decision System",
    page_icon="🏦",
    layout="wide"
)

# ======================================================
# LOAD MODEL & FEATURES
# ======================================================
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style='text-align:center; color:#0A1F44;'>🏦 Smart Loan Decision System</h1>
<p style='text-align:center; color:gray;'>
AI-powered, confidence-aware loan approval with human-in-the-loop review
</p>
<hr>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR — APPLICANT DETAILS
# ======================================================
st.sidebar.markdown("""
<h2 style='color:#0A1F44;'>👤 Applicant Details</h2>
<p style='color:gray;'>Enter customer financial information</p>
<hr>
""", unsafe_allow_html=True)

income = st.sidebar.number_input("Annual Income (₹)", min_value=0, step=50000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, step=50000)
loan_term = st.sidebar.slider("Loan Term (months)", 12, 360, 180)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0)

education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Employment Type", ["No", "Yes"])

# ======================================================
# BUILD INPUT DATA (MATCH TRAINING FEATURES)
# ======================================================
input_data = pd.DataFrame([{
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "no_of_dependents": dependents,
    "education_Not_Graduate": 1 if education == "Not Graduate" else 0,
    "self_employed_Yes": 1 if self_employed == "Yes" else 0,
}])

# Align input exactly with training features
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# ======================================================
# MAIN PANEL — DECISION OUTPUT
# ======================================================
st.markdown("## 📊 Loan Decision Outcome")
decision_box = st.container()

# ======================================================
# PREDICTION + FINAL DECISION LOGIC
# ======================================================
if st.button("🔍 Evaluate Loan Application"):
    prob = model.predict_proba(input_data)[0]
    prediction = np.argmax(prob)
    confidence = np.max(prob)

    LOW_CONF = 0.6
    HIGH_CONF = 0.8

    with decision_box:

        # 1️⃣ HARD BUSINESS REJECTIONS
        if cibil_score < 450:
            st.error("❌ **Loan Rejected**\n\nVery low credit score.")

        elif income > 0 and loan_amount > 5 * income:
            st.error("❌ **Loan Rejected**\n\nExtreme loan burden.")

        # 2️⃣ ML CONFIDENCE-BASED DECISIONS
        elif confidence < LOW_CONF:
            st.error("❌ **Loan Rejected**\n\nLow model confidence indicates high risk.")

        elif confidence < HIGH_CONF:
            st.warning("⚠️ **Manual Review Required**\n\nBorderline model confidence.")

        # 3️⃣ POLICY-BASED MANUAL REVIEW
        elif income > 0 and loan_amount > 1.2 * income:
            st.info("🧾 **Manual Review Required**\n\nHigh loan burden flagged by policy.")

        # 4️⃣ FINAL DECISION
        else:
            if prediction == 1:
                st.success("✅ **Loan Approved**\n\nCustomer meets approval criteria.")
            else:
                st.error("❌ **Loan Rejected**")

        st.metric("Model Confidence", f"{confidence:.2f}")

# ======================================================
# EXPLANATION SECTION
# ======================================================
with st.expander("ℹ️ How does the system decide?"):
    st.markdown("""
- **Hard rules** reject extremely risky cases immediately  
- **Low model confidence** → automatic rejection  
- **Borderline confidence** → human review  
- **High confidence + policy checks** → final decision  

This design ensures safe, explainable, and scalable credit decisions.
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 • AI-powered Credit Risk Prototype • Built ny Khrisha")
