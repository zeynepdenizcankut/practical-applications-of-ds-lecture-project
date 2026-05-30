"""
data_ingestion.py
-----------------
Fetches adverse drug event reports from the openFDA FAERS API
for GLP-1 receptor agonists (Semaglutide & Tirzepatide).

Uses limit=1000 (max allowed) for fast pagination.
Optionally uses an API key via the API_KEY environment variable
to bypass rate limits.

Docs: https://open.fda.gov/apis/drug/event/
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.fda.gov/drug/event.json"

# All GLP-1 brand and generic names to maximize coverage
DRUG_NAMES = [
    "SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS",
    "TIRZEPATIDE", "MOUNJARO", "ZEPBOUND",
]

# Active ingredient names for classifying the target drug
GENERICS = ["SEMAGLUTIDE", "TIRZEPATIDE"]

DATE_START = "20200101"
DATE_END = datetime.now().strftime("%Y%m%d")

PAGE_SIZE = 1000          # Max allowed by openFDA
TOTAL_TARGET = 30000     
MAX_RETRIES = 3
RETRY_DELAY = 3


def build_search_query():
    """Build a single combined search query for all GLP-1 drugs."""
    drug_clauses = " OR ".join([f'"{d}"' for d in DRUG_NAMES])
    query = (
        f"patient.drug.medicinalproduct:({drug_clauses})"
        f" AND receivedate:[{DATE_START} TO {DATE_END}]"
    )
    return query


def extract_record(report):
    """
    Flatten a single FAERS report into a dictionary.
    Handles nested and missing fields gracefully.
    """
    patient = report.get("patient", {})
    drugs = patient.get("drug", [])
    reactions = patient.get("reaction", [])

    # --- Identify the target GLP-1 drug vs. concomitant ---
    target_drug = {}
    concomitant_drugs = []
    for d in drugs:
        product_name = d.get("medicinalproduct", "").upper()
        generic_via_openfda = (
            d.get("openfda", {}).get("generic_name", [""])[0].upper()
            if d.get("openfda")
            else ""
        )
        combined = product_name + " " + generic_via_openfda

        if any(name in combined for name in DRUG_NAMES):
            target_drug = d
        else:
            concomitant_drugs.append(d)

    # --- Extract target drug fields ---
    brand = (
        target_drug.get("openfda", {}).get("brand_name", ["Unknown"])[0]
        if target_drug.get("openfda")
        else target_drug.get("medicinalproduct", "Unknown")
    )

    generic_name = "Unknown"
    if target_drug.get("openfda"):
        generic_name = target_drug["openfda"].get("generic_name", ["Unknown"])[0]
    else:
        # Infer from product name
        product_upper = target_drug.get("medicinalproduct", "").upper()
        for g in GENERICS:
            if g in product_upper:
                generic_name = g
                break
        # Also map brand → generic
        brand_map = {
            "OZEMPIC": "SEMAGLUTIDE", "WEGOVY": "SEMAGLUTIDE",
            "RYBELSUS": "SEMAGLUTIDE", "MOUNJARO": "TIRZEPATIDE",
            "ZEPBOUND": "TIRZEPATIDE",
        }
        for brand_key, gen in brand_map.items():
            if brand_key in product_upper:
                generic_name = gen
                break

    route = target_drug.get("drugadministrationroute", "Unknown")
    indication = target_drug.get("drugindication", "Unknown")
    dose = target_drug.get("drugdosagetext", "Unknown")
    drug_char = target_drug.get("drugcharacterization", "Unknown")

    # --- Reactions ---
    reaction_list = [r.get("reactionmeddrapt", "") for r in reactions]
    reaction_outcomes = [r.get("reactionoutcome", "") for r in reactions]

    # --- Patient ---
    age = patient.get("patientonsetage", None)
    age_unit = patient.get("patientonsetageunit", None)
    sex = patient.get("patientsex", None)
    weight = patient.get("patientweight", None)

    # --- Report-level seriousness ---
    return {
        "report_id": report.get("safetyreportid", None),
        "receive_date": report.get("receivedate", None),
        "country": report.get("occurcountry", "Unknown"),
        "report_type": report.get("reporttype", None),
        "serious": report.get("serious", None),
        "serious_death": report.get("seriousnessdeath", "0"),
        "serious_hospitalization": report.get("seriousnesshospitalization", "0"),
        "serious_disabling": report.get("seriousnessdisabling", "0"),
        "serious_lifethreatening": report.get("seriousnesslifethreatening", "0"),
        "serious_other": report.get("seriousnessother", "0"),
        "patient_age": age,
        "patient_age_unit": age_unit,
        "patient_sex": sex,
        "patient_weight": weight,
        "generic_name": generic_name,
        "brand_name": brand,
        "route": route,
        "indication": indication,
        "dose": dose,
        "drug_characterization": drug_char,
        "num_concomitant_drugs": len(concomitant_drugs),
        "concomitant_drug_names": "; ".join(
            d.get("medicinalproduct", "Unknown")
            for d in concomitant_drugs[:10]
        ),
        "num_reactions": len(reaction_list),
        "reactions": "; ".join(reaction_list),
        "reaction_outcomes": "; ".join(str(o) for o in reaction_outcomes),
    }


def run_ingestion(output_path="data/raw_adverse_events.csv"):
    """
    Main ingestion pipeline.
    Fetches all GLP-1 adverse events in a single paginated loop
    using limit=1000 for speed. Uses API_KEY if available.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("API_KEY")
    search_query = build_search_query()

    print(f"{'='*60}")
    print(f"openFDA FAERS Ingestion")
    print(f"  Date range: {DATE_START} → {DATE_END}")
    print(f"  API key:    {'provided' if api_key else 'not set (may be rate-limited)'}")
    print(f"{'='*60}")

    all_records = []
    start_time = time.time()

    for skip in range(0, TOTAL_TARGET, PAGE_SIZE):
        params = {
            "search": search_query,
            "limit": PAGE_SIZE,
            "skip": skip,
        }
        if api_key:
            params["api_key"] = api_key

        # --- Retry loop ---
        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 429:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code == 404:
                    # No more results beyond this skip
                    data = None
                    break
                else:
                    print(f"  HTTP {resp.status_code} – retrying ({attempt+1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e} – retrying ({attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)

        if data is None:
            print(f"  No more results at skip={skip}. Stopping.")
            break

        results = data.get("results", [])
        if not results:
            print(f"  Empty results at skip={skip}. Stopping.")
            break

        # Show total available on first page
        if skip == 0:
            total_available = data.get("meta", {}).get("results", {}).get("total", 0)
            print(f"  Total available on FDA: {total_available:,}")

        for report in results:
            all_records.append(extract_record(report))

        elapsed = time.time() - start_time
        print(f"  Fetched {len(all_records):,} records  ({elapsed:.0f}s elapsed)")

        # Short delay to be respectful (API key holders can go faster)
        time.sleep(0.5 if api_key else 1.0)
    print("Api key: ", api_key)
    # --- Deduplicate and save ---
    df = pd.DataFrame(all_records)
    before = len(df)
    df.drop_duplicates(subset=["report_id"], inplace=True)

    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Ingestion complete in {elapsed:.0f}s")
    print(f"  Raw records:    {before:,}")
    print(f"  After dedup:    {len(df):,}")
    print(f"  Saved to:       {output_path}")
    print(f"{'='*60}")
    return df


if __name__ == "__main__":
    run_ingestion()