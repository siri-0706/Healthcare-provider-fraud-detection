"""
MODELS PAGE — paste this into the block in your app.py that runs when
the "Models" nav option is selected (replacing your current placeholder table).
"""

import streamlit as st
import pandas as pd

st.title("🏥 Healthcare Provider Fraud Detection")
st.header("Machine Learning Models")

st.write(
    "Three models were trained and compared using stratified train/test splitting "
    "and class weighting to handle the 9.35% fraud base rate. The primary evaluation "
    "metrics were **F1-score** and **ROC-AUC** rather than accuracy, since accuracy is "
    "trivially high and uninformative on such a skewed target."
)

# --- Actual performance results from the notebook ---
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Purpose": ["Linear baseline (class-weighted)", "Tree ensemble (class-weighted)", "Boosting model (scale_pos_weight)"],
    "F1-score": [0.559, 0.667, 0.623],
    "ROC-AUC": [0.960, 0.955, 0.956],
})

st.dataframe(
    results_df.style.format({"F1-score": "{:.3f}", "ROC-AUC": "{:.3f}"}),
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(results_df.set_index("Model")[["F1-score", "ROC-AUC"]])

st.success(
    "**Random Forest was selected as the final model** — it gave the best F1-score "
    "(0.667) with a strong ROC-AUC (0.955). Five-fold stratified cross-validation "
    "confirmed the result was stable: **F1 = 0.61 ± 0.03, ROC-AUC = 0.93 ± 0.01** across folds."
)

st.subheader("Model performance in detail")
col1, col2, col3 = st.columns(3)
col1.metric("Recall (Fraud class)", "75%")
col2.metric("Precision (Fraud class)", "60%")
col3.metric("F1-optimal threshold", "0.56")

st.write(
    "At the default 0.5 probability threshold, the model catches 75% of fraudulent "
    "providers (recall) while keeping 60% precision. Threshold tuning found an "
    "**F1-optimal cut-off of ~0.56**, which balances recall and precision — recommended "
    "for production use over the default 0.5 (see Business Recommendations)."
)

st.subheader("Top predictive features")
importances_df = pd.DataFrame({
    "Feature": ["TotalReimbursed", "TotalDeductible", "MaxReimbursed", "N_Claims_IP",
                "MeanAdmissionDuration", "MeanNProcCodes", "StdReimbursed"],
    "Relative importance": [0.136, 0.106, 0.078, 0.068, 0.053, 0.045, 0.041],
})
st.dataframe(importances_df, use_container_width=True, hide_index=True)
st.caption(
    "Full ranking includes 43 engineered features across volume, physician-network, "
    "financial, duration, coding-complexity, and patient-population groups. "
    "See the notebook for the complete feature importance chart."
)
