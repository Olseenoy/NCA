# ================================
# File: src/rca_engine.py
# ================================
import json
import re
from datetime import datetime
import pandas as pd
from src.llm_rca import generate_rca_with_llm, LLMRCAException, extract_issue_with_source

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]


# -------------------------------
# Utilities
# -------------------------------
def generate_fishbone_skeleton() -> dict:
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def convert_to_fishbone(root_causes):
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone


# -------------------------------
# Build context
# -------------------------------
def build_context(issue_text: str, processed_df=None, sop_library=None, qc_logs=None) -> str:
    parsed = {"date": None, "shift": None, "machine": None, "lanes": [], "time_range": None, "defect": None}

    date_match = re.search(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", issue_text)
    if date_match:
        try:
            parsed["date"] = str(datetime.strptime(date_match.group(), "%Y-%m-%d").date())
        except Exception:
            parsed["date"] = date_match.group()

    machine_match = re.search(r"\bmc\s*\d+\b", issue_text, re.I)
    if machine_match:
        parsed["machine"] = machine_match.group()

    context = f"Raw issue: {issue_text}\n\nParsed context: {json.dumps(parsed, indent=2)}\n"
    if sop_library:
        context += f"\nRelevant SOPs:\n{sop_library}\n"
    if qc_logs:
        context += f"\nHistorical QC logs:\n{qc_logs}\n"
    if processed_df is not None:
        context += f"\nProcessed dataset snapshot (rows={len(processed_df)}):\n{processed_df.head(3).to_dict()}\n"

    return context


# -------------------------------
# RCA Orchestrator
# -------------------------------
def ai_rca_with_fallback(record: dict, processed_df=None, sop_library=None, qc_logs=None) -> dict:
    try:
        issue_text, source_col = extract_issue_with_source(record)
        if not issue_text:
            return {"error": "No valid issue text found.", "fishbone": generate_fishbone_skeleton()}

        context_text = build_context(issue_text, processed_df, sop_library, qc_logs)
        context_text += f"\n\n[System note: Issue text was extracted from column '{source_col}']"

        raw_result = generate_rca_with_llm(issue_text=issue_text, context=context_text)

        root_causes = raw_result.get("root_causes", [])
        return {
            "root_causes": root_causes,
            "five_whys": raw_result.get("five_whys", []),
            "capa": raw_result.get("capa", []),
            "fishbone": convert_to_fishbone(root_causes),
            "confidence": raw_result.get("confidence", "medium")
        }

    except LLMRCAException:
        return {"error": "LLM unavailable, fallback used.", "fishbone": generate_fishbone_skeleton()}
    except Exception as e:
        return {"error": f"RCA Engine Failure: {e}", "fishbone": generate_fishbone_skeleton()}


# -------------------------------
# Extract recurring issues
# -------------------------------
def extract_recurring_issues(df: pd.DataFrame, col_name_candidates=None, top_n: int = 10) -> dict:
    """
    Extract the most frequent issues from logs.
    Tries synonyms of 'issue' as column names.
    """
    if col_name_candidates is None:
        col_name_candidates = ["issue_description", "issue", "problem", "error", "failure", "incident"]

    issue_col = None
    for c in col_name_candidates:
        if c in df.columns:
            issue_col = c
            break

    if not issue_col:
        return {}

    issues = df[issue_col].dropna().astype(str)
    freq = issues.value_counts().head(top_n).to_dict()
    return freq


# -------------------------------
# Process uploaded SOPs / Docs
# -------------------------------
def process_uploaded_docs(uploaded_docs) -> str:
    """
    Convert uploaded SOPs or maintenance docs into plain text.
    Supports TXT, DOCX, and PDF.
    """
    texts = []
    for f in uploaded_docs:
        name = f.name.lower()

        if name.endswith(".txt"):
            texts.append(f.read().decode("utf-8", errors="ignore"))

        elif name.endswith(".docx"):
            import docx
            doc = docx.Document(f)
            texts.append("\n".join([p.text for p in doc.paragraphs]))

        elif name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(f)
            texts.append("\n".join([page.extract_text() for page in reader.pages if page.extract_text()]))

    return "\n\n".join(texts)

