import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Healthcare Provider Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main{
    padding-top:1rem;
}
.title{
    text-align:center;
    color:#0E76A8;
}
.metric{
    background-color:#f5f5f5;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏥 Healthcare Provider Fraud Detection")

st.markdown("""
### Machine Learning Based Healthcare Fraud Detection

This application demonstrates the complete workflow for detecting fraudulent healthcare providers using Machine Learning.
""")

st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Exploratory Data Analysis",
        "🤖 Machine Learning Models",
        "📋 Prediction Results",
        "💼 Business Recommendations"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.header("Project Overview")

    st.write("""
Healthcare fraud causes billions of dollars in losses every year.
This project aims to identify fraudulent healthcare providers using
historical Medicare claims data.
""")

    st.subheader("Project Objectives")

    st.markdown("""
- Detect fraudulent healthcare providers.
- Perform Exploratory Data Analysis.
- Engineer meaningful features.
- Train multiple Machine Learning models.
- Compare model performances.
- Predict fraud on unseen provider data.
""")

    st.subheader("Workflow")

    st.markdown("""
1. Data Collection

2. Data Cleaning

3. Exploratory Data Analysis

4. Feature Engineering

5. Feature Selection

6. Model Building

7. Prediction

8. Business Recommendations
""")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Models Used", "3")

    with c2:
        st.metric("Prediction File", "1353 Providers")

    with c3:
        st.metric("Task", "Binary Classification")
        # ==================================================
# DATASET OVERVIEW
# ==================================================

elif page == "📊 Dataset Overview":

    st.header("Dataset Overview")

    st.write("""
The fraud detection system was developed using multiple Medicare claim datasets.
These datasets were merged and transformed into provider-level features for
training machine learning models.
""")

    dataset = pd.DataFrame({
        "Dataset": [
            "Beneficiary Data",
            "Inpatient Claims",
            "Outpatient Claims",
            "Provider Labels",
            "Unseen Provider Data"
        ],
        "Description": [
            "Patient demographic and chronic disease information",
            "Hospital admission claim records",
            "Outpatient medical claim records",
            "Fraud / Non-Fraud provider labels",
            "Providers for final prediction"
        ]
    })

    st.dataframe(dataset, use_container_width=True)

    st.subheader("Data Preprocessing")

    st.markdown("""
- Removed duplicate records
- Handled missing values
- Merged inpatient and outpatient claims
- Aggregated claims at provider level
- Converted categorical variables
- Created numerical features for model training
""")

    st.subheader("Feature Engineering")

    features = pd.DataFrame({
        "Feature": [
            "Total Claims",
            "Average Claim Amount",
            "Total Reimbursement",
            "Average Inpatient Duration",
            "Number of Beneficiaries",
            "Chronic Disease Count"
        ],
        "Purpose": [
            "Claim frequency",
            "Financial behaviour",
            "Provider reimbursement",
            "Hospital stay analysis",
            "Provider workload",
            "Patient health profile"
        ]
    })

    st.table(features)


# ==================================================
# EXPLORATORY DATA ANALYSIS
# ==================================================

elif page == "📈 Exploratory Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write("""
Exploratory Data Analysis (EDA) was performed to understand claim patterns,
provider behaviour and fraud distribution before model building.
""")

    st.subheader("Key Observations")

    st.markdown("""
- Fraudulent providers generally submit a higher number of claims.
- Higher reimbursement amounts indicate increased fraud risk.
- Longer inpatient stays are associated with suspicious providers.
- Providers with many beneficiaries require closer monitoring.
- Chronic disease patterns influence claim frequency.
""")

    st.subheader("Example Claim Distribution")

    chart = pd.DataFrame({
        "Category": [
            "Inpatient",
            "Outpatient",
            "Fraud",
            "Non-Fraud"
        ],
        "Count": [
            42000,
            98000,
            506,
            847
        ]
    })

    st.bar_chart(chart.set_index("Category"))

    st.subheader("Insights")

    st.info("""
The EDA revealed strong differences in claim behaviour between fraudulent
and non-fraudulent providers, making machine learning an effective approach
for fraud detection.
""")
    # ==================================================
# MACHINE LEARNING MODELS
# ==================================================

elif page == "🤖 Machine Learning Models":

    st.header("Machine Learning Models")

    st.write("""
Three machine learning models were trained and evaluated for healthcare
provider fraud detection.
""")

    models = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Description": [
            "Baseline linear classifier",
            "Ensemble learning using decision trees",
            "Gradient boosting classifier"
        ]
    })

    st.dataframe(models, use_container_width=True)

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Advantages": [
            "Simple and interpretable",
            "Handles complex relationships",
            "High predictive performance"
        ]
    })

    st.table(comparison)

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models Evaluated", "3")

    with col2:
        st.metric("Prediction Task", "Fraud Detection")

    with col3:
        st.metric("Output", "Fraud / Non-Fraud")

    st.info("""
Random Forest and XGBoost generally perform better than Logistic Regression
because they capture complex non-linear relationships in healthcare claim data.
""")

    st.subheader("Example Feature Importance")

    feature_importance = pd.DataFrame({
        "Feature": [
            "Total Claims",
            "Total Reimbursement",
            "Average Claim Amount",
            "Beneficiary Count",
            "Hospital Stay",
            "Chronic Disease Count"
        ],
        "Importance": [
            0.30,
            0.24,
            0.18,
            0.12,
            0.10,
            0.06
        ]
    })

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")

    st.pyplot(fig)

    st.success("""
The most influential features were Total Claims, Total Reimbursement,
and Average Claim Amount.
""")
    # ==================================================
# PREDICTION RESULTS
# ==================================================

elif page == "📋 Prediction Results":

    st.header("Prediction Results")

    st.write("""
The trained model was applied to the unseen provider dataset.
The output contains the predicted fraud class and probability for each provider.
""")

    csv_file = "Siri Chandana Submission.csv"

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)

        st.success("Prediction file loaded successfully.")

        st.subheader("Prediction Sample")

        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Dataset Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Providers", len(df))

        if "PredictedClass" in df.columns:

            fraud = (df["PredictedClass"] == "Yes").sum()
            nonfraud = (df["PredictedClass"] == "No").sum()

            with c2:
                st.metric("Fraud Providers", fraud)

            with c3:
                st.metric("Non-Fraud Providers", nonfraud)

            st.subheader("Fraud Prediction Distribution")

            prediction_count = df["PredictedClass"].value_counts()

            st.bar_chart(prediction_count)

            st.subheader("Prediction Percentage")

            percentage = (
                prediction_count / prediction_count.sum() * 100
            ).round(2)

            st.dataframe(
                percentage.rename("Percentage (%)"),
                use_container_width=True
            )

        if "Probability" in df.columns:

            st.subheader("Fraud Probability Distribution")

            fig, ax = plt.subplots(figsize=(8,4))

            ax.hist(df["Probability"], bins=20)

            ax.set_xlabel("Fraud Probability")

            ax.set_ylabel("Providers")

            ax.set_title("Distribution of Fraud Probability")

            st.pyplot(fig)

        st.download_button(
            label="Download Prediction CSV",
            data=df.to_csv(index=False),
            file_name="Siri Chandana Submission.csv",
            mime="text/csv"
        )

    else:

        st.error("""
Prediction file not found.

Please place 'Siri Chandana Submission.csv'
in the same folder as app.py.
""")
        # ==================================================
# BUSINESS RECOMMENDATIONS
# ==================================================

elif page == "💼 Business Recommendations":

    st.header("Business Recommendations")

    st.write("""
Based on the fraud detection analysis, the following recommendations can help
healthcare organizations reduce fraudulent claims and improve claim monitoring.
""")

    recommendations = [
        "Monitor providers with unusually high claim volumes.",
        "Track providers with abnormal reimbursement amounts.",
        "Flag providers showing sudden increases in claim frequency.",
        "Perform periodic audits for high-risk providers.",
        "Use machine learning models during claim approval.",
        "Continuously retrain models with new healthcare data.",
        "Combine fraud prediction with manual investigation for high-risk cases."
    ]

    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f"**{i}. {rec}**")

    st.subheader("Benefits")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
- Reduced fraudulent payments
- Faster fraud detection
- Improved claim monitoring
- Better resource utilization
""")

    with col2:
        st.info("""
- Data-driven decision making
- Early fraud identification
- Increased operational efficiency
- Lower financial losses
""")

    st.subheader("Project Conclusion")

    st.write("""
This project demonstrates how machine learning can be used to identify
potentially fraudulent healthcare providers by analyzing historical Medicare
claims data. After preprocessing, feature engineering, and model evaluation,
the trained model successfully generated predictions for unseen providers.

Such predictive systems can support healthcare organizations in detecting
suspicious providers earlier, reducing financial losses, and improving the
overall efficiency of fraud investigation.
""")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">
        <h4>🏥 Healthcare Provider Fraud Detection</h4>
        <p>Developed using Python, Machine Learning, Pandas, Scikit-learn, XGBoost and Streamlit</p>
        <p>© 2026 Siri Chandana</p>
    </div>
    """,
    unsafe_allow_html=True,
)
        

  


         
