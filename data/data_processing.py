"""
data_processing.py
------------------
Cleans raw FAERS data and engineers features for modeling.

Key steps:
  1. Parse and normalize dates
  2. Standardize age to years
  3. Map categorical codes to readable labels
  4. Engineer polypharmacy and reaction-burden features
  5. Create the binary target variable: is_serious
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ----- Mapping dictionaries based on FAERS documentation -----

SEX_MAP = {"0": "Unknown", "1": "Male", "2": "Female"}

AGE_UNIT_MAP = {
    "800": "decade",
    "801": "year",
    "802": "month",
    "803": "week",
    "804": "day",
    "805": "hour",
}

AGE_TO_YEARS = {
    "decade": 10.0,
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 1.0 / 52.0,
    "day": 1.0 / 365.25,
    "hour": 1.0 / 8766.0,
}

REPORT_TYPE_MAP = {
    "1": "Expedited (15-day)",
    "2": "Periodic",
    "3": "Direct",
    "4": "Other",
}

ROUTE_SIMPLIFY = {
    "Subcutaneous": "Subcutaneous",
    "Oral": "Oral",
    "Intravenous": "Intravenous",
    "Intramuscular": "Intramuscular",
}


def load_raw_data(path="data/raw_adverse_events.csv"):
    """Load the raw CSV from the ingestion step."""
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} raw records from {path}")
    return df


def parse_dates(df):
    """Convert receivedate (YYYYMMDD string) to datetime."""
    df["receive_date"] = pd.to_datetime(
        df["receive_date"], format="%Y%m%d", errors="coerce"
    )
    df["report_year"] = df["receive_date"].dt.year
    df["report_quarter"] = df["receive_date"].dt.quarter
    df["report_month"] = df["receive_date"].dt.month
    return df


def standardize_age(df):
    """
    Convert patient_age to years using the age_unit field.
    Clamp to a plausible human range [0, 120].
    """
    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
    df["patient_age_unit"] = (
        pd.to_numeric(df["patient_age_unit"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .map(AGE_UNIT_MAP)
    )

    multiplier = df["patient_age_unit"].map(AGE_TO_YEARS).fillna(1.0)
    df["age_years"] = df["patient_age"] * multiplier

    # Clamp unrealistic values
    df.loc[df["age_years"] < 0, "age_years"] = np.nan
    df.loc[df["age_years"] > 120, "age_years"] = np.nan

    return df


def map_sex(df):
    """Map sex codes to readable labels."""
    df["patient_sex"] = (
        pd.to_numeric(df["patient_sex"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .map(SEX_MAP)
        .fillna("Unknown")
    )
    return df

def map_report_type(df):
    """Map report type codes."""
    df["report_type"] = (
        pd.to_numeric(df["report_type"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .map(REPORT_TYPE_MAP)
        .fillna("Unknown")
    )
    return df


def standardize_weight(df):
    """Clean weight (assumed kg). Clamp to [20, 350]."""
    df["patient_weight"] = pd.to_numeric(df["patient_weight"], errors="coerce")
    df.loc[df["patient_weight"] < 20, "patient_weight"] = np.nan
    df.loc[df["patient_weight"] > 350, "patient_weight"] = np.nan
    return df


def simplify_route(df):
    """Simplify administration route to common categories."""
    df["route"] = df["route"].apply(
        lambda x: ROUTE_SIMPLIFY.get(x, "Other") if pd.notna(x) else "Other"
    )
    return df


def simplify_indication(df):
    """Group indications into major categories."""
    def _categorize(text):
        if pd.isna(text) or text == "Unknown":
            return "Unknown"
        text = str(text).upper()
        if any(k in text for k in ["DIABET", "GLYC", "A1C", "INSULIN", "TYPE 2"]):
            return "Diabetes"
        elif any(k in text for k in ["WEIGHT", "OBES", "BMI", "OVERWEIGHT"]):
            return "Weight Management"
        elif any(k in text for k in ["HEART", "CARDIO", "HYPERT"]):
            return "Cardiovascular"
        else:
            return "Other"

    df["indication_group"] = df["indication"].apply(_categorize)
    return df


def create_target(df):
    """
    Build binary target: is_serious.
    A report is 'serious' if the serious field is '1' OR
    any sub-seriousness flag is '1'.
    """
    seriousness_cols = [
        "serious_death",
        "serious_hospitalization",
        "serious_disabling",
        "serious_lifethreatening",
        "serious_other",
    ]

    for col in seriousness_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["serious"] = pd.to_numeric(df["serious"], errors="coerce").fillna(0).astype(int)
    df["is_serious"] = (
        (df["serious"] == 1) | (df[seriousness_cols].sum(axis=1) > 0)
    ).astype(int)

    return df


def engineer_features(df):
    """
    Create derived features for modeling:
      - polypharmacy_flag: patient takes >= 3 other drugs
      - reaction_burden: number of reported reactions
      - bmi_proxy: rough BMI if weight and age are available
      - age_bin: discretized age groups
    """
    # Polypharmacy
    df["num_concomitant_drugs"] = pd.to_numeric(
        df["num_concomitant_drugs"], errors="coerce"
    ).fillna(0).astype(int)
    df["polypharmacy_flag"] = (df["num_concomitant_drugs"] >= 3).astype(int)

    # Reaction burden
    df["num_reactions"] = pd.to_numeric(
        df["num_reactions"], errors="coerce"
    ).fillna(0).astype(int)

    # Age bins
    bins = [0, 30, 45, 60, 75, 120]
    labels = ["<30", "30-44", "45-59", "60-74", "75+"]
    df["age_bin"] = pd.cut(df["age_years"], bins=bins, labels=labels, right=False)
    df["age_bin"] = df["age_bin"].cat.add_categories("Unknown").fillna("Unknown")

    return df


def select_modeling_columns(df):
    """Select and return only the columns needed downstream."""
    keep_cols = [
        # identifiers
        "report_id", "receive_date", "report_year", "report_quarter", "report_month",
        # patient
        "age_years", "age_bin", "patient_sex", "patient_weight",
        # drug
        "generic_name", "brand_name", "route", "indication_group",
        "drug_characterization",
        # polypharmacy
        "num_concomitant_drugs", "polypharmacy_flag",
        "concomitant_drug_names",
        # reactions
        "num_reactions", "reactions",
        # seriousness
        "serious_death", "serious_hospitalization",
        "serious_disabling", "serious_lifethreatening", "serious_other",
        "is_serious",
        # geography & meta
        "country", "report_type",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def run_processing(
    input_path="data/raw_adverse_events.csv",
    output_path="data/processed_adverse_events.csv",
):
    """Full processing pipeline."""
    df = load_raw_data(input_path)

    df = parse_dates(df)
    df = standardize_age(df)
    df = map_sex(df)
    df = map_report_type(df)
    df = standardize_weight(df)
    df = simplify_route(df)
    df = simplify_indication(df)
    df = create_target(df)
    df = engineer_features(df)
    df = select_modeling_columns(df)

    # Drop rows with no report_id
    df = df.dropna(subset=["report_id"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nProcessing complete.")
    print(f"  Records: {len(df):,}")
    print(f"  Serious: {df['is_serious'].sum():,}  ({df['is_serious'].mean():.1%})")
    print(f"  Saved to: {output_path}")
    return df


if __name__ == "__main__":
    run_processing()
