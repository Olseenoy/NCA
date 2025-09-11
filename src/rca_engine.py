# ================================
# File: src/rca_engine.py
# ================================
import json
import re
from datetime import datetime
from src.llm_rca import generate_rca_with_llm, LLMRCAException, extract_issue_with_source

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]


def generate_fishbone_skeleton() -> dict:
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def convert_to_fishbone(root_causes):
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone


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
