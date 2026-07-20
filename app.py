import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Healthcare Provider Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Provider Fraud Detection")
st.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Project Overview",
        "Dataset",
        "Models",
        "Predictions",
        "Business Recommendations"
    ]
)

if page == "Home":

    st.header("Welcome")

    st.write("""
This project predicts whether a healthcare provider is potentially fraudulent
using historical Medicare claim information.

The objective is to assist insurance companies and healthcare organizations
in identifying suspicious providers early so that detailed investigations can
be performed.
""")

    st.subheader("Project Highlights")

    st.markdown("""
- Machine Learning based fraud prediction
- Data preprocessing and feature engineering
- Multiple classification models
- Model evaluation
- Prediction on unseen providers
""")



elif page == "Project Overview":

    st.header("Project Overview")

    st.write("""
Healthcare fraud leads to significant financial losses every year.

The goal of this project is to classify healthcare providers into:

- Fraud
- Non-Fraud

using historical inpatient, outpatient and beneficiary information.
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


elif page == "Dataset":

    st.header("Dataset")

    st.write("""
The project uses multiple datasets including:

- Beneficiary Data
- Inpatient Claims
- Outpatient Claims
- Provider Labels
- Unseen Provider Data
""")

    st.subheader("Feature Engineering")

    st.markdown("""
Examples of engineered features include:

- Total Claims
- Total Reimbursement
- Chronic Disease Count
- Number of Beneficiaries
- Average Claim Amount
- Average Inpatient Duration
""")


elif page == "Models":

    st.header("Machine Learning Models")

    data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Purpose": [
            "Baseline",
            "Tree Ensemble",
            "Boosting Model"
        ]
    })

    st.table(data)

    st.success(
        "Random Forest and XGBoost generally perform better for fraud detection because they capture complex relationships."
    )


elif page == "Predictions":

    st.header("Prediction Results")

    csv_file = "Siri Chandana Submission.csv"

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)

        st.success("Prediction file loaded successfully.")

        st.write(df.head())

        if "PredictedClass" in df.columns:

            st.subheader("Prediction Counts")

            st.bar_chart(df["PredictedClass"].value_counts())

        st.download_button(
            "Download Prediction CSV",
            data=df.to_csv(index=False),
            file_name="Siri Chandana Submission.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Prediction CSV not found.\n\n"
            "Upload 'Siri Chandana Submission.csv' to the repository."
        )


elif page == "Business Recommendations":

    st.header("Business Recommendations")

    st.markdown("""
### Recommendations

- Monitor providers with unusually high claim amounts.

- Track abnormal inpatient and outpatient claim patterns.

- Perform periodic fraud audits.

- Use predictive analytics before approving high-risk claims.

- Continuously retrain fraud detection models using new data.

- Integrate ML models into claim processing systems for real-time fraud detection.
""")

st.markdown("---")
st.caption("Healthcare Provider Fraud Detection | Machine Learning Project")
