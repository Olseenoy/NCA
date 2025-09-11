# ==================================================================
# File: src/rca_engine.py
# ==================================================================
"""
RCA engine: preprocessing, document loading, context building, and orchestrator helpers.
Drop into src/rca_engine.py
"""
import re
import json
import pandas as pd
from datetime import datetime
from typing import Optional, List

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
# Document loaders (lightweight)
# -------------------------------

def process_uploaded_docs(uploaded_docs: Optional[List]) -> str:
    """Return a combined plain-text string from uploaded PDF/DOCX/TXT files."""
    try:
        from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    except Exception:
        # Minimal fallback: read bytes and decode as text for txt files
        texts = []
        if not uploaded_docs:
            return ""
        for doc in uploaded_docs:
            try:
                if doc.name.endswith('.txt'):
                    raw = doc.getvalue().decode('utf-8', errors='ignore')
                    texts.append(raw)
                else:
                    texts.append(f"[Unsupported file for reading in fallback: {doc.name}]")
            except Exception as e:
                texts.append(f"[Failed to read {doc.name}: {e}]")
        return "\n\n".join(texts)

    docs_content = []
    if uploaded_docs:
        for doc in uploaded_docs:
            try:
                if doc.name.endswith('.pdf'):
                    docs = PyPDFLoader(doc).load()
                elif doc.name.endswith('.docx'):
                    docs = Docx2txtLoader(doc).load()
                elif doc.name.endswith('.txt'):
                    docs = TextLoader(doc).load()
                else:
                    docs = [f"[Unsupported file type: {doc.name}]"]
                for d in docs:
                    # LangChain returns Document objects with .page_content
                    if hasattr(d, 'page_content'):
                        docs_content.append(d.page_content)
                    else:
                        docs_content.append(str(d))
            except Exception as e:
                docs_content.append(f"[Failed to load {doc.name}: {e}]")
    return "\n\n".join(docs_content)

# -------------------------------
# Context Builder
# -------------------------------

def build_context(issue_text: str, processed_df: Optional[pd.DataFrame] = None, sop_library: str = "", qc_logs: str = "") -> str:
    parsed = {
        "date": None,
        "shift": None,
        "machine": None,
        "lanes": [],
        "time_range": None,
        "defect": None,
    }

    # Date
    date_match = re.search(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", issue_text)
    if date_match:
        try:
            parsed["date"] = str(datetime.strptime(date_match.group(), "%Y-%m-%d").date())
        except Exception:
            parsed["date"] = date_match.group()

    # Time range
    time_match = re.search(r"(\d{1,2}:\d{2}\s?(?:am|pm)?-\d{1,2}:\d{2}\s?(?:am|pm)?)", issue_text, re.I)
    if time_match:
        parsed["time_range"] = time_match.group()

    # Machine
    machine_match = re.search(r"\bmc\s*\d+\b", issue_text, re.I)
    if machine_match:
        parsed["machine"] = machine_match.group()

    # Lanes
    lanes_match = re.findall(r"\blane[s]?\s*[:=]?\s*([\d, ]+)", issue_text, re.I)
    if lanes_match:
        parsed["lanes"] = [x.strip() for x in lanes_match[0].split(",")]

    # Defect
    defect_match = re.search(r"(poor|leak|contamination|misalignment|defect|perforation)", issue_text, re.I)
    if defect_match:
        parsed["defect"] = defect_match.group()

    context = f"Raw issue: {issue_text}\n\nParsed context: {json.dumps(parsed, indent=2)}\n"

    if sop_library:
        context += f"\nRelevant SOP snippets:\n{sop_library}\n"
    if qc_logs:
        context += f"\nHistorical QC logs (recent):\n{qc_logs}\n"
    if processed_df is not None and isinstance(processed_df, pd.DataFrame):
        context += f"\nProcessed dataset snapshot (rows={len(processed_df)}):\n{processed_df.head(3).to_dict()}\n"

    return context

# -------------------------------
# Recurring Issues Extractor
# -------------------------------

def extract_recurring_issues(logs_df: pd.DataFrame, col_name: str = "issue_description", top_n: int = 5):
    if col_name not in logs_df.columns:
        return {}
    return logs_df[col_name].value_counts().head(top_n).to_dict()

# -------------------------------
# Rule-Based RCA
# -------------------------------

def rule_based_rca_suggestions(clean_text: str) -> dict:
    keywords_map = {
        'operator': 'Man', 'training': 'Man', 'calibration': 'Measurement',
        'machine': 'Machine', 'overheat': 'Machine', 'contamination': 'Material',
        'label': 'Method', 'procedure': 'Method', 'ambient': 'Environment',
        'lane': 'Machine', 'sealing': 'Machine', 'sterilization': 'Method'
    }
    fishbone = generate_fishbone_skeleton()
    for kw, cat in keywords_map.items():
        if kw.lower() in clean_text.lower():
            fishbone[cat].append(kw)
    return fishbone

# -------------------------------
# Orchestrator (keeps backward compatibility)
# -------------------------------

def ai_rca_with_fallback(original_text: str, clean_text: str, processed_df=None, sop_library=None, qc_logs=None) -> dict:
    """Build context and call the LLM RCA generator. Returns standard RCA dict."""
    try:
        context_text = build_context(original_text, processed_df, sop_library or "", qc_logs or "")
        # Delegate to llm_rca.generate_rca_with_llm which handles LLM selection & fallbacks
        result = generate_rca_with_llm(issue_text=original_text, context=context_text)

        # If the result is string attempt JSON parse
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return {"error": "LLM returned invalid JSON.", "fishbone": rule_based_rca_suggestions(original_text)}

        root_causes = result.get("root_causes") or []
        five_whys_chain = result.get("five_whys") or []
        capa_actions = result.get("capa") or []

        return {
            "analysis": result.get("analysis", ""),
            "root_causes": root_causes,
            "five_whys": five_whys_chain,
            "capa": capa_actions,
            "fishbone": convert_to_fishbone(root_causes),
            "confidence": result.get("confidence", "medium")
        }

    except Exception as e:
        return {"error": f"RCA Engine Failure: {e}", "fishbone": rule_based_rca_suggestions(original_text)}
