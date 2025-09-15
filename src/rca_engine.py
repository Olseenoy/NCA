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

# Optional: for SOP parsing
try:
    import docx2txt
    import PyPDF2
except ImportError:
    docx2txt = None
    PyPDF2 = None


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
    except Exception as e:
        print(f"[ERROR] Failed to load logs: {e}")
        return None


def load_sop_documents(folder: str) -> str:
    """
    Load SOP/manual documents (supports txt, pdf, docx).
    Returns concatenated text for LLM context.
    """
    sop_texts = []
    for file in glob.glob(os.path.join(folder, "*.*")):
        try:
            if file.endswith(".txt"):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    sop_texts.append(f.read())

            elif file.endswith(".docx") and docx2txt:
                sop_texts.append(docx2txt.process(file))

            elif file.endswith(".pdf") and PyPDF2:
                text = []
                with open(file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text.append(page.extract_text() or "")
                sop_texts.append("\n".join(text))

        except Exception as e:
            print(f"[WARNING] Could not parse {file}: {e}")

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
    """
    Run RCA analysis using LLM (Ollama) or fallback rule-based method.
    Returns dict with analysis, causes, 5-whys, capa, and fishbone visualization.
    """
    try:
        if mode == "AI-Powered (LLM+Agent)":
            result = call_llm(issue_text, sop_text)
        else:
            # Rule-based fallback
            result = {
                "analysis": f"Rule-based RCA for: {issue_text}",
                "root_causes": ["Generic cause A", "Generic cause B"],
                "five_whys": ["Why 1", "Why 2", "Why 3", "Why 4", "Why 5"],
                "capa": [
                    {"type": "Corrective", "action": "Check X", "owner": "QA", "due_in_days": 5}
                ],
                "fishbone": {
                    "Man": ["N/A"],
                    "Machine": ["N/A"],
                    "Method": ["N/A"],
                    "Material": ["N/A"],
                    "Environment": ["N/A"],
                    "Measurement": ["N/A"],
                },
            }

        # --- Build Fishbone Visualization ---
        if "fishbone" in result and result["fishbone"]:
            categories, causes = [], []
            for k, vals in result["fishbone"].items():
                for v in vals:
                    categories.append(k)
                    causes.append(v)

            if categories:
                fig = px.scatter(
                    x=categories,
                    y=causes,
                    title="Fishbone Diagram",
                    labels={"x": "Category", "y": "Cause"},
                )
                result["fishbone_fig"] = fig

        return result

    except Exception as e:
        return {"error": f"RCA failed: {e}"}

