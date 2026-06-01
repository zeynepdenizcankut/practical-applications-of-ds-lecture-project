"""
eda.py
------
Exploratory Data Analysis functions for the Streamlit dashboard.
Each function returns a Plotly figure object for interactive rendering.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = ["#0077B6", "#00B4D8", "#90E0EF", "#F77F00", "#D62828", "#FCBF49"]


def plot_severity_distribution(df):
    """Donut chart showing serious vs. non-serious reports."""
    counts = df["is_serious"].value_counts().reset_index()
    counts.columns = ["is_serious", "count"]
    counts["label"] = counts["is_serious"].map({1: "Serious", 0: "Non-Serious"})

    fig = px.pie(
        counts,
        values="count",
        names="label",
        color="label",
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
        hole=0.5,
    )
    fig.update_layout(
        title="Severity Distribution of Adverse Events",
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=50, b=30),
    )
    return fig


def plot_reports_over_time(df):
    """Monthly time series of report counts by severity."""
    df_ts = df.copy()
    df_ts["month"] = df_ts["receive_date"].dt.to_period("M").astype(str)
    grouped = (
        df_ts.groupby(["month", "is_serious"])
        .size()
        .reset_index(name="count")
    )
    grouped["Severity"] = grouped["is_serious"].map({1: "Serious", 0: "Non-Serious"})

    fig = px.area(
        grouped,
        x="month",
        y="count",
        color="Severity",
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
    )
    fig.update_layout(
        title="Monthly Adverse Event Reports Over Time",
        xaxis_title="Month",
        yaxis_title="Report Count",
        font=dict(size=13),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    fig.update_xaxes(dtick="M3", tickangle=45)
    return fig


def plot_age_distribution(df):
    """Histogram of patient age colored by severity."""
    df_age = df.dropna(subset=["age_years"]).copy()
    df_age["Severity"] = df_age["is_serious"].map({1: "Serious", 0: "Non-Serious"})

    fig = px.histogram(
        df_age,
        x="age_years",
        color="Severity",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
    )
    fig.update_layout(
        title="Age Distribution by Severity",
        xaxis_title="Patient Age (years)",
        yaxis_title="Count",
        font=dict(size=13),
        margin=dict(t=50, b=30),
    )
    return fig


def plot_sex_severity(df):
    """Grouped bar chart: severity breakdown by sex."""
    grouped = (
        df.groupby(["patient_sex", "is_serious"])
        .size()
        .reset_index(name="count")
    )
    grouped["Severity"] = grouped["is_serious"].map({1: "Serious", 0: "Non-Serious"})

    fig = px.bar(
        grouped,
        x="patient_sex",
        y="count",
        color="Severity",
        barmode="group",
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
    )
    fig.update_layout(
        title="Severity by Patient Sex",
        xaxis_title="Sex",
        yaxis_title="Count",
        font=dict(size=13),
        margin=dict(t=50, b=30),
    )
    return fig


def plot_drug_comparison(df):
    """Compare severity rates between Semaglutide and Tirzepatide."""
    drug_sev = (
        df.groupby("generic_name")["is_serious"]
        .agg(["sum", "count"])
        .reset_index()
    )
    drug_sev.columns = ["Drug", "Serious", "Total"]
    drug_sev["Non-Serious"] = drug_sev["Total"] - drug_sev["Serious"]
    drug_sev["Serious Rate (%)"] = (drug_sev["Serious"] / drug_sev["Total"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Serious",
        x=drug_sev["Drug"],
        y=drug_sev["Serious"],
        marker_color="#D62828",
        text=drug_sev["Serious Rate (%)"].astype(str) + "%",
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Non-Serious",
        x=drug_sev["Drug"],
        y=drug_sev["Non-Serious"],
        marker_color="#0077B6",
    ))
    fig.update_layout(
        barmode="stack",
        title="Drug Comparison: Severity Breakdown",
        yaxis_title="Report Count",
        font=dict(size=13),
        margin=dict(t=50, b=30),
    )
    return fig


def plot_polypharmacy_effect(df):
    """Show how number of concomitant drugs relates to severity."""
    df_poly = df.copy()
    df_poly["concomitant_bin"] = pd.cut(
        df_poly["num_concomitant_drugs"],
        bins=[-1, 0, 2, 5, 10, 100],
        labels=["0", "1-2", "3-5", "6-10", ">10"],
    )
    grouped = (
        df_poly.groupby("concomitant_bin", observed=False)["is_serious"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped.columns = ["Concomitant Drugs", "Serious Rate", "N"]

    fig = px.bar(
        grouped,
        x="Concomitant Drugs",
        y="Serious Rate",
        text=grouped["Serious Rate"].apply(lambda x: f"{x:.1%}"),
        color="Serious Rate",
        color_continuous_scale=["#0077B6", "#F77F00", "#D62828"],
    )
    fig.update_layout(
        title="Polypharmacy Effect: Seriousness Rate by Number of Concomitant Drugs",
        yaxis_title="Proportion Serious",
        yaxis_tickformat=".0%",
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=30),
    )
    fig.update_coloraxes(showscale=False)
    return fig


def plot_top_reactions(df, top_n=15):
    """Horizontal bar chart of the most frequently reported reactions."""
    all_reactions = df["reactions"].dropna().str.split("; ").explode()
    top = all_reactions.value_counts().head(top_n).reset_index()
    top.columns = ["Reaction", "Count"]

    fig = px.bar(
        top,
        x="Count",
        y="Reaction",
        orientation="h",
        color="Count",
        color_continuous_scale=["#90E0EF", "#0077B6", "#023E8A"],
    )
    fig.update_layout(
        title=f"Top {top_n} Reported Adverse Reactions",
        yaxis=dict(autorange="reversed"),
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=30, l=180),
    )
    fig.update_coloraxes(showscale=False)
    return fig


def plot_indication_severity(df):
    """Seriousness rate by indication group."""
    grouped = (
        df.groupby("indication_group")["is_serious"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped.columns = ["Indication", "Serious Rate", "N"]
    grouped = grouped[grouped["N"] >= 10].sort_values("Serious Rate", ascending=True)

    fig = px.bar(
        grouped,
        x="Serious Rate",
        y="Indication",
        orientation="h",
        text=grouped["Serious Rate"].apply(lambda x: f"{x:.1%}"),
        color="Serious Rate",
        color_continuous_scale=["#0077B6", "#F77F00", "#D62828"],
    )
    fig.update_layout(
        title="Seriousness Rate by Indication Group",
        xaxis_tickformat=".0%",
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=30, l=150),
    )
    fig.update_coloraxes(showscale=False)
    return fig


def plot_country_map(df, top_n=15):
    """Bar chart of top reporting countries."""
    top_countries = df["country"].value_counts().head(top_n).reset_index()
    top_countries.columns = ["Country", "Reports"]

    fig = px.bar(
        top_countries,
        x="Reports",
        y="Country",
        orientation="h",
        color="Reports",
        color_continuous_scale=["#90E0EF", "#0077B6"],
    )
    fig.update_layout(
        title=f"Top {top_n} Reporting Countries",
        yaxis=dict(autorange="reversed"),
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=30, l=50),
    )
    fig.update_coloraxes(showscale=False)
    return fig


def plot_weight_severity(df):
    """Box plot of patient weight by severity."""
    df_w = df.dropna(subset=["patient_weight"]).copy()
    df_w["Severity"] = df_w["is_serious"].map({1: "Serious", 0: "Non-Serious"})

    fig = px.box(
        df_w,
        x="Severity",
        y="patient_weight",
        color="Severity",
        color_discrete_map={"Serious": "#D62828", "Non-Serious": "#0077B6"},
    )
    fig.update_layout(
        title="Patient Weight Distribution by Severity",
        yaxis_title="Weight (kg)",
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=30),
    )
    return fig
