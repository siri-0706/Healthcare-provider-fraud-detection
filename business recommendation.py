"""
BUSINESS RECOMMENDATIONS PAGE — paste this into the block in your app.py that
runs when the "Business Recommendations" nav option is selected (replacing
your current generic bullet list).
"""

import streamlit as st

st.title("🏥 Healthcare Provider Fraud Detection")
st.header("Business Recommendations")

st.subheader("How the model should be used")
st.markdown("""
- **Deploy as a triage / prioritisation tool** for a Special Investigations Unit (SIU),
  not as an automatic denial mechanism — flagged providers should be ranked by predicted
  probability so investigators work the highest-risk cases first.
- **Use the F1-optimised threshold (~0.56)** rather than the default 0.5 to favour recall,
  since missing a fraudulent provider is typically costlier than one extra manual review.
""")

st.subheader("Operational red flags surfaced by the model")
st.markdown("""
- Unusually high **total or maximum claim reimbursement** relative to a provider's peer
  group and patient count.
- Unusually **long inpatient admission durations**, or a high inpatient share of total claims.
- **High procedure/diagnosis code counts per claim** (possible upcoding).
- A high share of claims where **the same physician is listed as both attending and
  operating** (identity reuse / lack of independent oversight).
""")

st.subheader("Suggested improvements for a future iteration")
st.markdown("""
- **Physician-level and network features:** build a provider–physician–beneficiary graph
  and add graph features (e.g. shared beneficiaries across providers) to better capture
  collusion rings.
- **Peer-group normalisation:** compare each provider's financials against similarly
  sized/specialised/regional providers to reduce false positives for large, legitimately
  high-volume providers.
- **Diagnosis/procedure code embeddings:** encode which specific codes are used (e.g.
  frequency or target encoding) rather than only counting them, to capture specific
  billing-pattern anomalies.
- **Cost-sensitive threshold selection:** calibrate the decision threshold against the
  SIU's actual investigation cost versus average fraud loss once those figures are available.
- **Periodic retraining:** fraud patterns evolve, so the model should be revalidated
  against new investigation outcomes on a regular (e.g. quarterly) cycle.
""")

st.subheader("Limitations")
st.warning("""
- All features are derived from claims and beneficiary data alone; no direct evidence of
  fraud (e.g. investigation outcomes) is available, so the model learns correlates of the
  historical label rather than causal proof of fraud.
- The training label set is relatively small (506 positive cases out of 5,410 providers),
  which limits how much the model can learn about rarer fraud patterns; performance
  should be monitored as more investigation outcomes become available.
""")
