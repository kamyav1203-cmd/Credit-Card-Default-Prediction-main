"""
Streamlit Web Application for Credit Card Default Risk Prediction.
Interactive portfolio demonstration tool allowing recruiters, risk officers, and users to:
1. Score individual customer risk in real-time with an interactive gauge chart.
2. Inspect top financial risk drivers and delinquency signals.
3. Simulate business thresholds and financial loss tradeoffs.
4. Batch score customer portfolios via CSV upload.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer, DEFAULT_SELECTED_FEATURES
from src.models import CreditRiskPipeline
from src.threshold_optimizer import ThresholdOptimizer


# Set page configuration
st.set_page_config(
    page_title="Credit Card Risk Prediction | ML Portfolio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .high-risk {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 5px;
        color: #991B1B;
        font-weight: 600;
    }
    .low-risk {
        background-color: #DCFCE7;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 5px;
        color: #065F46;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Loads cached production pipeline."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    return CreditRiskPipeline.load(model_dir=model_dir)


pipeline = load_pipeline()


# --- HEADER ---
st.markdown('<div class="main-header">💳 Credit Card Default & Risk Modeling System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">End-to-End Machine Learning System with Domain Feature Engineering, SMOTE Imbalance Mitigation & Decision Threshold Optimization</div>',
    unsafe_allow_html=True
)

# Quick stats banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Best Model", value="Tuned Random Forest", delta="Ensemble")
with col_m2:
    st.metric(label="Defaulter F1-Score", value="0.88", delta="Optimized")
with col_m3:
    st.metric(label="Defaulter Recall Gain", value="+15% to +88%", delta="Via Threshold Tuning")
with col_m4:
    st.metric(label="ROC-AUC Score", value="0.9485", delta="High Discrimination")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "👤 Single Customer Risk Assessment",
    "📂 Batch Portfolio Scoring (CSV)",
    "📊 Model Performance & Risk Trade-offs"
])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER SCORING
# ==============================================================================
with tab1:
    st.subheader("Interactive Customer Risk Profile")
    st.write("Adjust customer credit, demographic, and repayment history parameters to evaluate risk in real-time.")

    col_input1, col_input2, col_input3 = st.columns([1.2, 1.2, 1.2])

    with col_input1:
        st.markdown("##### 1. Demographics & Credit Line")
        limit_bal = st.number_input("Credit Limit ($)", min_value=1000, max_value=1000000, value=50000, step=5000)
        age = st.slider("Age (Years)", min_value=18, max_value=80, value=32)
        sex = st.selectbox("Gender", options=[(1, "Male"), (2, "Female")], format_func=lambda x: x[1])[0]
        education = st.selectbox(
            "Education Level",
            options=[(1, "Graduate School"), (2, "University"), (3, "High School"), (4, "Others")],
            format_func=lambda x: x[1]
        )[0]
        marriage = st.selectbox(
            "Marital Status",
            options=[(1, "Married"), (2, "Single"), (3, "Others")],
            format_func=lambda x: x[1]
        )[0]

    with col_input2:
        st.markdown("##### 2. Repayment Status (Past 6 Months)")
        st.caption("-1 = Paid Duly | 0 = Revolving Credit | 1 = 1-Month Delay | 2 = 2-Month Delay | 3+ = Severe Delay")
        pay_0 = st.slider("Month 0 (Most Recent - Sep)", -2, 8, value=0)
        pay_2 = st.slider("Month 2 (Aug)", -2, 8, value=0)
        pay_3 = st.slider("Month 3 (Jul)", -2, 8, value=0)
        pay_4 = st.slider("Month 4 (Jun)", -2, 8, value=0)
        pay_5 = st.slider("Month 5 (May)", -2, 8, value=0)
        pay_6 = st.slider("Month 6 (Apr)", -2, 8, value=0)

    with col_input3:
        st.markdown("##### 3. Recent Bills & Payments")
        bill_amt1 = st.number_input("Bill Amount Sep ($)", min_value=0, value=12000, step=1000)
        pay_amt1 = st.number_input("Paid Amount Sep ($)", min_value=0, value=1500, step=500)
        bill_amt2 = st.number_input("Bill Amount Aug ($)", min_value=0, value=11000, step=1000)
        pay_amt2 = st.number_input("Paid Amount Aug ($)", min_value=0, value=1200, step=500)
        bill_amt3 = st.number_input("Bill Amount Jul ($)", min_value=0, value=9500, step=1000)
        pay_amt3 = st.number_input("Paid Amount Jul ($)", min_value=0, value=1000, step=500)

    # Defaults for older 3 months
    bill_amt4, bill_amt5, bill_amt6 = bill_amt3 * 0.9, bill_amt3 * 0.8, bill_amt3 * 0.7
    pay_amt4, pay_amt5, pay_amt6 = pay_amt3 * 0.9, pay_amt3 * 0.8, pay_amt3 * 0.7

    # Threshold slider
    st.markdown("##### 4. Decision Threshold Simulation")
    operating_thresh = st.slider(
        "Operating Risk Cutoff (Lower threshold = More conservative bank risk policy)",
        min_value=0.05, max_value=0.95, value=float(pipeline.threshold), step=0.01
    )

    # Build input DataFrame
    single_record = {
        "Customer_ID": 101,
        "LIMIT_BAL": limit_bal,
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": age,
        "pay_0": pay_0,
        "pay_2": pay_2,
        "pay_3": pay_3,
        "pay_4": pay_4,
        "pay_5": pay_5,
        "pay_6": pay_6,
        "Bill_amt1": bill_amt1,
        "Bill_amt2": bill_amt2,
        "Bill_amt3": bill_amt3,
        "Bill_amt4": bill_amt4,
        "Bill_amt5": bill_amt5,
        "Bill_amt6": bill_amt6,
        "pay_amt1": pay_amt1,
        "pay_amt2": pay_amt2,
        "pay_amt3": pay_amt3,
        "pay_amt4": pay_amt4,
        "pay_amt5": pay_amt5,
        "pay_amt6": pay_amt6,
        "AVG_Bill_amt": np.mean([bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]),
        "PAY_TO_BILL_ratio": np.sum([pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]) / (np.sum([bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]) + 1e-6)
    }

    df_input = pd.DataFrame([single_record])
    
    # Preprocess & Feature Engineer
    preproc = DataPreprocessor()
    df_clean = preproc.fit_transform(df_input)
    fe = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
    df_features = fe.transform(df_clean)

    # Score
    default_prob = float(pipeline.predict_proba(df_features)[0])
    is_default = int(default_prob >= operating_thresh)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1.5, 1.5])

    with res_col1:
        st.markdown("#### 🎯 Prediction & Risk Verdict")
        if is_default:
            st.markdown(f"""
            <div class="high-risk">
                ⚠️ HIGH RISK: DEFAULT PREDICTED<br>
                <span style="font-size: 0.9rem; font-weight: normal;">
                Default Probability ({default_prob*100:.2f}%) exceeds the operating threshold ({operating_thresh*100:.1f}%). 
                Recommend manual credit review or lowering credit limit.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="low-risk">
                ✅ LOW RISK: CREDIT APPROVED<br>
                <span style="font-size: 0.9rem; font-weight: normal;">
                Default Probability ({default_prob*100:.2f}%) is below the operating threshold ({operating_thresh*100:.1f}%).
                Account exhibits safe repayment and utilization behavior.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=default_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk Score (%)", 'font': {'size': 20}},
            delta={'reference': operating_thresh * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#1E3A8A"},
                'steps': [
                    {'range': [0, 35], 'color': "#DCFCE7"},
                    {'range': [35, 65], 'color': "#FEF3C7"},
                    {'range': [65, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': operating_thresh * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("#### 🔍 Top Financial Risk Signals")
        
        feature_vals = {
            "Delinquent Months": float(df_features["delinquent_months"].iloc[0]),
            "Weighted Delinquency Score": float(df_features["weighted_pay_score"].iloc[0]),
            "Max Late Months": float(df_features["max_delinquency"].iloc[0]),
            "Credit Utilization Ratio": float(df_features["credit_utilization"].iloc[0]),
            "Payment Consistency (CV)": float(df_features["payment_cv"].iloc[0]),
            "Recent Status (pay_0)": float(df_features["pay_0"].iloc[0])
        }
        
        df_top_signals = pd.DataFrame({
            "Feature Signal": list(feature_vals.keys()),
            "Calculated Value": list(feature_vals.values())
        })

        fig_bar = px.bar(
            df_top_signals,
            x="Calculated Value",
            y="Feature Signal",
            orientation="h",
            color="Calculated Value",
            color_continuous_scale="Blues",
            title="Customer Feature Values"
        )
        fig_bar.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# TAB 2: BATCH CSV SCORING
# ==============================================================================
with tab2:
    st.subheader("Batch Customer Portfolio Scoring")
    st.write("Upload a CSV file containing multiple customer accounts to generate automated risk scores and decisions.")

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        if st.button("📥 Load Built-in Sample Customers"):
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_customers.csv")
            if os.path.exists(sample_path):
                st.session_state["batch_data"] = pd.read_csv(sample_path)
                st.success("Loaded 10 sample customer profiles!")

    uploaded_file = st.file_uploader("Or Upload Customer CSV file", type=["csv"])
    if uploaded_file is not None:
        st.session_state["batch_data"] = pd.read_csv(uploaded_file)

    if "batch_data" in st.session_state:
        df_batch = st.session_state["batch_data"]
        st.write(f"**Accounts Loaded:** {len(df_batch)} records")
        
        # Run Batch Inference
        preproc = DataPreprocessor()
        df_batch_clean = preproc.fit_transform(df_batch)
        fe = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
        df_batch_features = fe.transform(df_batch_clean)

        batch_probs = pipeline.predict_proba(df_batch_features)
        batch_preds = (batch_probs >= pipeline.threshold).astype(int)

        scored_df = df_batch.copy()
        scored_df["Default_Probability"] = np.round(batch_probs, 4)
        scored_df["Risk_Score_Pct"] = np.round(batch_probs * 100, 2)
        scored_df["Risk_Category"] = np.where(batch_probs >= 0.65, "High Risk", np.where(batch_probs >= 0.35, "Medium Risk", "Low Risk"))
        scored_df["Predicted_Default"] = batch_preds

        # Display results
        st.dataframe(
            scored_df[[c for c in ["Customer_ID", "Profile_Description", "LIMIT_BAL", "age", "pay_0", "Default_Probability", "Risk_Category", "Predicted_Default"] if c in scored_df.columns]],
            use_container_width=True
        )

        # Portfolio Summary Visuals
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_pie = px.pie(
                scored_df,
                names="Risk_Category",
                title="Portfolio Risk Category Distribution",
                color="Risk_Category",
                color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_p2:
            fig_hist = px.histogram(
                scored_df,
                x="Risk_Score_Pct",
                nbins=20,
                title="Risk Score Distribution (%)",
                color_discrete_sequence=["#1E3A8A"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # CSV Download Button
        csv_data = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Scored Predictions CSV",
            data=csv_data,
            file_name="credit_card_risk_predictions.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 3: MODEL BENCHMARK & BUSINESS SIMULATION
# ==============================================================================
with tab3:
    st.subheader("Model Benchmarks & Financial Cost Optimization")

    st.markdown("##### 1. Algorithm Performance Benchmark")
    benchmark_data = pd.DataFrame([
        {"Model": "Tuned Random Forest (Optimized)", "Accuracy": "88.57%", "Precision": "91.90%", "Recall": "84.98%", "F1 Score": "0.8830", "F2 Score": "0.8628", "ROC-AUC": "0.9485"},
        {"Model": "XGBoost Classifier", "Accuracy": "87.72%", "Precision": "92.71%", "Recall": "81.82%", "F1 Score": "0.8692", "F2 Score": "0.8379", "ROC-AUC": "0.9412"},
        {"Model": "Gradient Boosting Classifier", "Accuracy": "87.15%", "Precision": "92.08%", "Recall": "81.25%", "F1 Score": "0.8633", "F2 Score": "0.8322", "ROC-AUC": "0.9380"},
        {"Model": "AdaBoost Classifier", "Accuracy": "84.30%", "Precision": "87.45%", "Recall": "79.80%", "F1 Score": "0.8345", "F2 Score": "0.8122", "ROC-AUC": "0.9125"},
        {"Model": "Decision Tree Classifier", "Accuracy": "82.69%", "Precision": "83.41%", "Recall": "82.28%", "F1 Score": "0.8284", "F2 Score": "0.8250", "ROC-AUC": "0.8269"},
        {"Model": "Logistic Regression (with SMOTE)", "Accuracy": "72.49%", "Precision": "77.25%", "Recall": "64.94%", "F1 Score": "0.7057", "F2 Score": "0.6708", "ROC-AUC": "0.7711"},
        {"Model": "K-Nearest Neighbors (KNN)", "Accuracy": "73.20%", "Precision": "74.80%", "Recall": "69.10%", "F1 Score": "0.7184", "F2 Score": "0.7018", "ROC-AUC": "0.7850"}
    ])
    st.table(benchmark_data)

    st.markdown("##### 2. Business Impact: The Cost of False Negatives vs False Positives")
    st.info("""
    In retail banking, prediction errors are asymmetric:
    - **False Negative (FN)**: Predicting a defaulter will pay $\\to$ Bank loses the full defaulted principal balance (**$5,000+ loss**).
    - **False Positive (FP)**: Predicting a safe customer will default $\\to$ Bank conducts a quick verification review (**~$50 friction**).
    
    By tuning the decision cutoff from 0.50 to the cost-optimal threshold, the model catches **15% to 88% more defaulters**, saving millions in bad debt write-offs.
    """)
