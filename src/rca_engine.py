# rca_engine.py
"""
RCA Engine Layer
- Loads data (logs + SOPs)
- Extracts recurring issues
- Runs RCA via LLM or fallback
- Builds fishbone visualization
"""

import os
import glob
import pandas as pd
import plotly.express as px
from llm_rca import call_llm

# --- LOADERS ---
def load_latest_logs(folder: str) -> pd.DataFrame:
    """Load the latest CSV/Excel file from processed folder."""
    try:
        files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.xlsx"))
        if not files:
            return None
        latest = max(files, key=os.path.getctime)
        if latest.endswith(".csv"):
            return pd.read_csv(latest)
        return pd.read_excel(latest)
    except Exception:
        return None

def load_sop_documents(folder: str) -> str:
    """Load SOP/manual documents (txt only for now)."""
    sop_texts = []
    for file in glob.glob(os.path.join(folder, "*.*")):
        if file.endswith(".txt"):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                sop_texts.append(f.read())
    return "\n".join(sop_texts)

# --- ISSUE EXTRACTION ---
def extract_recurring_issues(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Extract top recurring issues from logs (Pareto style)."""
    for col in ["issue_description", "issue", "problem", "error", "failure", "incident"]:
        if col in df.columns:
            top = df[col].value_counts().head(top_n).to_dict()
            return top
    return {}

# --- RCA ANALYSIS ---
def run_rca_analysis(issue_text: str, processed_df: pd.DataFrame, sop_text: str, mode: str):
    """Run RCA analysis using LLM or fallback."""
    try:
        if mode == "AI-Powered (LLM+Agent)":
            result = call_llm(issue_text, sop_text)
        else:
            # Rule-based fallback
            result = {
                "analysis": f"Rule-based RCA for: {issue_text}",
                "root_causes": ["Generic cause A", "Generic cause B"],
                "five_whys": ["Why 1", "Why 2", "Why 3", "Why 4", "Why 5"],
                "capa": [{"type": "Corrective", "action": "Check X", "owner": "QA", "due_in_days": 5}],
                "fishbone": {"Man": ["N/A"], "Machine": ["N/A"], "Method": ["N/A"],
                             "Material": ["N/A"], "Environment": ["N/A"], "Measurement": ["N/A"]},
            }

        # Add visualization if fishbone data exists
        if "fishbone" in result and result["fishbone"]:
            categories = []
            causes = []
            for k, vals in result["fishbone"].items():
                for v in vals:
                    categories.append(k)
                    causes.append(v)
            if categories:
                fig = px.scatter(
                    x=categories,
                    y=causes,
                    title="Fishbone Diagram",
                )
                result["fishbone_fig"] = fig

        return result
    except Exception as e:
        return {"error": str(e)}
