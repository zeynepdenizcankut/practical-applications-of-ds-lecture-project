"""
app.py
------
Streamlit dashboard for the Pharmacovigilance Severity Predictor.

Pages:
  1. Overview — KPIs and project description
  2. Exploratory Analysis — interactive charts
  3. Model Performance — metrics, ROC, confusion matrix
  4. Predict — live severity prediction form
"""

import json
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.data_ingestion import run_ingestion
from data.data_processing import run_processing
from model.modeling import train_and_evaluate, prepare_features, FEATURE_COLS
from eda.eda import (
    plot_severity_distribution,
    plot_reports_over_time,
    plot_age_distribution,
    plot_sex_severity,
    plot_drug_comparison,
    plot_polypharmacy_effect,
    plot_top_reactions,
    plot_indication_severity,
    plot_country_map,
    plot_weight_severity,
)

# ─── Page Config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Pharmacovigilance · GLP-1 Adverse Event Severity Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');

    .stApp {
        font-family: 'Source Sans 3', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #023E8A 0%, #0077B6 50%, #00B4D8 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        font-size: 1.05rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    .kpi-card {
        background: #f8f9fa;
        border-left: 4px solid #0077B6;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
    }
    .kpi-card h3 {
        font-size: 2rem;
        font-weight: 700;
        color: #023E8A;
        margin: 0;
    }
    .kpi-card p {
        font-size: 0.85rem;
        color: #555;
        margin: 0.3rem 0 0 0;
    }
    .kpi-serious {
        border-left-color: #D62828;
    }
    .kpi-serious h3 {
        color: #D62828;
    }

    div[data-testid="stSidebar"] {
        background: #f0f4f8;
    }

    .prediction-result {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-serious {
        background: linear-gradient(135deg, #D62828, #F77F00);
        color: white;
    }
    .pred-nonserious {
        background: linear-gradient(135deg, #0077B6, #00B4D8);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Pipeline ──────────────────────────────────────────────

RAW_PATH = "data/raw_adverse_events.csv"
PROCESSED_PATH = "data/processed_adverse_events.csv"
MODEL_DIR = Path("models")


@st.cache_data(show_spinner="Fetching data from openFDA API...")
def load_or_fetch_data():
    """Run the full ingestion → processing → modeling pipeline once."""
    raw = Path(RAW_PATH)
    processed = Path(PROCESSED_PATH)
    model_file = MODEL_DIR / "best_model.joblib"

    if not raw.exists():
        run_ingestion(RAW_PATH)

    if not processed.exists():
        run_processing(RAW_PATH, PROCESSED_PATH)

    if not model_file.exists():
        train_and_evaluate(PROCESSED_PATH, str(MODEL_DIR))

    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    df["receive_date"] = pd.to_datetime(df["receive_date"], errors="coerce")
    return df


def load_model_artifacts():
    """Load saved model, scaler, encoders, and metrics."""
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    feat_imp = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    test_preds = pd.read_csv(MODEL_DIR / "test_predictions.csv")
    return model, scaler, encoders, metrics, feat_imp, test_preds


# ─── Sidebar ────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.favpng.com/4/6/23/clip-art-pharmacovigilance-pharmacy-pharmaceutical-industry-health-care-png-favpng-y1wD0Vd9CYBH7M7Xf8aPDGWdz.jpg", width=80)
    st.markdown("## Pharmacovigilance")
    st.markdown("**GLP-1 Adverse Event Severity Predictor**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🔮 Predict Severity"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Data: <a href='https://open.fda.gov/apis/drug/event/' target='_blank'>openFDA FAERS</a><br>"
        "Drugs: Semaglutide · Tirzepatide<br>"
        "Period: 2020 – 2023</small>",
        unsafe_allow_html=True,
    )


# ─── Load Data ──────────────────────────────────────────────────

df = load_or_fetch_data()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>💊 Pharmacovigilance — GLP-1 Adverse Event Severity Predictor</h1>
        <p>Predicting the severity of adverse drug events for Semaglutide (Ozempic/Wegovy) and Tirzepatide (Mounjaro) using FDA FAERS data.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total = len(df)
    serious_count = df["is_serious"].sum()
    serious_rate = df["is_serious"].mean()
    unique_drugs = df["generic_name"].nunique()
    countries = df["country"].nunique()
    avg_age = df["age_years"].median()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="kpi-card"><h3>{total:,}</h3><p>Total Reports</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card kpi-serious"><h3>{serious_count:,}</h3><p>Serious Events</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card kpi-serious"><h3>{serious_rate:.1%}</h3><p>Seriousness Rate</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><h3>{countries}</h3><p>Countries</p></div>', unsafe_allow_html=True)
    with c5:
        median_age_val = f"{avg_age:.0f}" if pd.notna(avg_age) else "N/A"
        st.markdown(f'<div class="kpi-card"><h3>{median_age_val}</h3><p>Median Age (yrs)</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("### About This Project")
        st.markdown("""
        **Pharmacovigilance (PV)** is the science of detecting and preventing adverse effects
        of medications. With the recent surge in GLP-1 receptor agonist prescriptions for diabetes
        and weight management, understanding their safety profiles is a critical public health task.

        This application:
        - **Fetches** real-time adverse event data from the [openFDA FAERS API](https://open.fda.gov/apis/drug/event/)
        - **Processes** and engineers features such as polypharmacy burden and patient demographics
        - **Trains** multiple ML models to predict whether a reported event will be classified as *serious*
        - **Visualizes** patterns through interactive dashboards
        - **Predicts** severity for new patient profiles in real-time
        """)

    with col_r:
        st.markdown("### Seriousness Criteria (FDA)")
        st.markdown("""
        An adverse event is **serious** if it results in:
        - 🏥 Hospitalization
        - ⚠️ Life-threatening condition
        - 🦽 Disability or incapacity
        - ☠️ Death
        - 🔬 Other medically important condition
        """)

    st.markdown("---")
    # Quick charts on overview
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_severity_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_drug_comparison(df), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: EXPLORATORY ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "📊 Exploratory Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Interactive exploration of adverse event patterns, demographics, and risk factors.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            drug_filter = st.multiselect(
                "Drug",
                options=df["generic_name"].dropna().unique().tolist(),
                default=df["generic_name"].dropna().unique().tolist(),
            )
        with fc2:
            sex_filter = st.multiselect(
                "Sex",
                options=df["patient_sex"].unique().tolist(),
                default=df["patient_sex"].unique().tolist(),
            )
        with fc3:
            indication_filter = st.multiselect(
                "Indication",
                options=df["indication_group"].unique().tolist(),
                default=df["indication_group"].unique().tolist(),
            )

    df_filtered = df[
        (df["generic_name"].isin(drug_filter))
        & (df["patient_sex"].isin(sex_filter))
        & (df["indication_group"].isin(indication_filter))
    ]

    st.caption(f"Showing **{len(df_filtered):,}** of {len(df):,} reports after filters.")

    # Charts
    st.plotly_chart(plot_reports_over_time(df_filtered), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_age_distribution(df_filtered), use_container_width=True)
    with col2:
        st.plotly_chart(plot_sex_severity(df_filtered), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_polypharmacy_effect(df_filtered), use_container_width=True)
    with col4:
        st.plotly_chart(plot_weight_severity(df_filtered), use_container_width=True)

    st.plotly_chart(plot_top_reactions(df_filtered), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_indication_severity(df_filtered), use_container_width=True)
    with col6:
        st.plotly_chart(plot_country_map(df_filtered), use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Performance & Evaluation</h1>
        <p>Comparison of ML models trained to predict adverse event severity.</p>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, encoders, metrics, feat_imp, test_preds = load_model_artifacts()

    best_name = metrics["best_model"]
    st.success(f"**Best Model: {best_name}**")

    # Model comparison table
    st.markdown("### Model Comparison")
    rows = []
    for name, res in metrics["results"].items():
        rows.append({
            "Model": name,
            "CV ROC-AUC": f"{res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}",
            "Test ROC-AUC": f"{res['test_roc_auc']:.4f}",
            "Test PR-AUC": f"{res['test_pr_auc']:.4f}",
            "Test F1": f"{res['test_f1']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ROC Curves
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ROC Curves")
        fig_roc = go.Figure()
        colors = {"Logistic Regression": "#00B4D8", "Random Forest": "#F77F00", "Gradient Boosting": "#D62828"}
        for name, res in metrics["results"].items():
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                name=f"{name} (AUC={res['test_roc_auc']:.3f})",
                line=dict(color=colors.get(name, "#333"), width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            font=dict(size=13),
            margin=dict(t=30, b=30),
            legend=dict(x=0.4, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown("### Precision-Recall Curves")
        fig_pr = go.Figure()
        for name, res in metrics["results"].items():
            fig_pr.add_trace(go.Scatter(
                x=res["recall_curve"], y=res["precision_curve"],
                name=f"{name} (AUC={res['test_pr_auc']:.3f})",
                line=dict(color=colors.get(name, "#333"), width=2),
            ))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            font=dict(size=13),
            margin=dict(t=30, b=30),
            legend=dict(x=0.0, y=0.1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Confusion Matrix for best model
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"### Confusion Matrix — {best_name}")
        cm = np.array(metrics["results"][best_name]["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Non-Serious", "Serious"],
            y=["Non-Serious", "Serious"],
            text_auto=True,
            color_continuous_scale=["#f0f4f8", "#023E8A"],
        )
        fig_cm.update_layout(font=dict(size=14), margin=dict(t=30, b=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        st.markdown("### Feature Importance")
        fig_fi = px.bar(
            feat_imp.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#90E0EF", "#0077B6", "#023E8A"],
        )
        fig_fi.update_layout(
            yaxis_title="",
            xaxis_title="Importance",
            font=dict(size=13),
            showlegend=False,
            margin=dict(t=30, b=30, l=150),
        )
        fig_fi.update_coloraxes(showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Prediction distribution
    st.markdown("### Test Set Predicted Probability Distribution")
    fig_dist = px.histogram(
        test_preds,
        x="y_proba",
        color=test_preds["y_true"].map({1: "Serious", 0: "Non-Serious"}),
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
        labels={"color": "Actual Class"},
    )
    fig_dist.update_layout(
        xaxis_title="Predicted Probability of Serious",
        yaxis_title="Count",
        font=dict(size=13),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4: PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "🔮 Predict Severity":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Predict Adverse Event Severity</h1>
        <p>Enter a patient profile to predict whether an adverse event would be classified as serious.</p>
    </div>
    """, unsafe_allow_html=True)

    model, scaler, encoders, metrics, feat_imp, _ = load_model_artifacts()

    st.markdown("### Patient & Medication Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Patient Age (years)", 18, 100, 55)
        weight = st.slider("Patient Weight (kg)", 40, 250, 95)
        sex = st.selectbox("Patient Sex", ["Female", "Male", "Unknown"])

    with col2:
        drug = st.selectbox("Medication", ["SEMAGLUTIDE", "TIRZEPATIDE"])
        route = st.selectbox("Administration Route", ["Subcutaneous", "Oral", "Other"])
        indication = st.selectbox(
            "Primary Indication",
            ["Diabetes", "Weight Management", "Cardiovascular", "Other"],
        )

    with col3:
        n_concomitant = st.slider("Number of Concomitant Drugs", 0, 20, 3)
        n_reactions = st.slider("Number of Reported Reactions", 1, 20, 2)
        polypharmacy = 1 if n_concomitant >= 3 else 0

    if st.button("🔍 Predict Severity", type="primary", use_container_width=True):
        # Encode inputs using stored label encoders
        sex_enc = encoders["sex_encoded"].transform([sex])[0] if sex in encoders["sex_encoded"].classes_ else 0
        drug_enc = encoders["drug_encoded"].transform([drug])[0] if drug in encoders["drug_encoded"].classes_ else 0
        route_enc = encoders["route_encoded"].transform([route])[0] if route in encoders["route_encoded"].classes_ else 0
        ind_enc = encoders["indication_encoded"].transform([indication])[0] if indication in encoders["indication_encoded"].classes_ else 0

        features = np.array([[
            age, weight, n_concomitant, n_reactions,
            polypharmacy, sex_enc, drug_enc, route_enc, ind_enc
        ]])

        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0][1]
        prediction = "Serious" if proba >= 0.5 else "Non-Serious"

        # Display result
        css_class = "pred-serious" if prediction == "Serious" else "pred-nonserious"
        icon = "⚠️" if prediction == "Serious" else "✅"

        st.markdown(f"""
        <div class="prediction-result {css_class}">
            <h2>{icon} Predicted: {prediction}</h2>
            <h3>Probability of Serious Outcome: {proba:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Risk factor breakdown
        st.markdown("### Risk Factor Contribution")
        feature_labels = [
            "Age", "Weight", "Concomitant Drugs", "Reaction Count",
            "Polypharmacy", "Sex", "Drug", "Route", "Indication"
        ]
        feature_values = features[0]

        # Show a gauge-style chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Seriousness Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#D62828" if proba >= 0.5 else "#0077B6"},
                "steps": [
                    {"range": [0, 30], "color": "#e8f4f8"},
                    {"range": [30, 60], "color": "#fef3e2"},
                    {"range": [60, 100], "color": "#fde2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            height=300,
            font=dict(size=14),
            margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Input summary table
        st.markdown("### Input Summary")
        input_df = pd.DataFrame({
            "Parameter": feature_labels,
            "Value": [
                f"{age} years", f"{weight} kg", str(n_concomitant),
                str(n_reactions), "Yes" if polypharmacy else "No",
                sex, drug, route, indication,
            ],
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
