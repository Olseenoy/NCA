# ================================
# File: src/rca_engine.py
# ================================
import re
import json
from datetime import datetime
from src.llm_rca import generate_rca_with_llm, LLMRCAException

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

# -------------------------------
# Utility Functions
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
# Context Builder
# -------------------------------
def build_context(issue_text: str, processed_df=None, sop_library=None, qc_logs=None) -> str:
    """
    Preprocess raw issue text + enrich with SOPs, QC logs, and historical data.
    """

    # Extract structured info from free text
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

    # Build context string
    context = f"Raw issue: {issue_text}\n\nParsed context: {json.dumps(parsed, indent=2)}\n"

    if sop_library:
        context += f"\nRelevant SOPs:\n{sop_library}\n"
    if qc_logs:
        context += f"\nHistorical QC logs (recent):\n{qc_logs}\n"
    if processed_df is not None:
        context += f"\nProcessed dataset snapshot (rows={len(processed_df)}):\n{processed_df.head(3).to_dict()}\n"

    return context

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
# RCA Orchestrator
# -------------------------------
def ai_rca_with_fallback(original_text: str, clean_text: str,
                         processed_df=None, sop_library=None, qc_logs=None) -> dict:
    try:
        # Build enriched context for the LLM
        context_text = build_context(original_text, processed_df, sop_library, qc_logs)

        raw_result = generate_rca_with_llm(issue_text=context_text)

        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                return {"error": "LLM returned invalid JSON.", "fishbone": rule_based_rca_suggestions(original_text)}
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            return {"error": "LLM returned unexpected type.", "fishbone": rule_based_rca_suggestions(original_text)}

        root_causes = result.get("root_causes") or []
        five_whys_chain = result.get("five_whys") or []
        capa_actions = result.get("capa") or []

        return {
            "root_causes": root_causes,
            "five_whys": five_whys_chain,
            "capa": capa_actions,
            "fishbone": convert_to_fishbone(root_causes),
            "confidence": result.get("confidence", "medium")
        }

    except LLMRCAException:
        return {
            "error": "LLM unavailable, used HuggingFace fallback.",
            **huggingface_rca(original_text)
        }
    except Exception as e:
        return {"error": f"RCA Engine Failure: {e}", "fishbone": rule_based_rca_suggestions(original_text)}

# -------------------------------
# HuggingFace Fallback (placeholder)
# -------------------------------
def huggingface_rca(issue_text: str) -> dict:
    return {
        "root_causes": [{"cause": "HF RCA cause", "category": "Method"}],
        "five_whys": ["HF Why1", "HF Why2", "HF Why3", "HF Why4", "HF Why5"],
        "capa": [
            {"type": "Corrective", "action": "HF corrective action", "owner": "QA Team", "due_in_days": 7},
            {"type": "Preventive", "action": "HF preventive action", "owner": "Maintenance", "due_in_days": 14}
        ],
        "fishbone": generate_fishbone_skeleton(),
        "confidence": "low"
    }
