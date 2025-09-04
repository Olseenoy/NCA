# ================================
# File: src/rca_engine.py
# ================================
import json
import pandas as pd
from src.llm_rca import generate_rca_with_llm, LLMRCAException

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

def rule_based_rca_suggestions(clean_text: str) -> dict:
    keywords_map = {
        'operator': 'Man', 'training': 'Man', 'calibration': 'Measurement',
        'machine': 'Machine', 'overheat': 'Machine', 'contamination': 'Material',
        'label': 'Method', 'procedure': 'Method', 'ambient': 'Environment'
    }
    fishbone = generate_fishbone_skeleton()
    for kw, cat in keywords_map.items():
        if kw.lower() in clean_text.lower():
            fishbone[cat].append(kw)
    return fishbone

def huggingface_rca(issue_text: str) -> dict:
    return {
        "root_causes": [{"cause": "HF RCA cause", "category": "Method"}],
        "five_whys": ["HF Why1", "HF Why2", "HF Why3", "HF Why4", "HF Why5"],
        "capa": [
            {"type": "Corrective", "action": "HF corrective action", "owner": "QA Team", "due_in_days": 7},
            {"type": "Preventive", "action": "HF preventive action", "owner": "Maintenance", "due_in_days": 14}
        ],
        "fishbone": generate_fishbone_skeleton()
    }

# -------------------------------
# Context Builder
# -------------------------------
def build_rca_context(issue_text: str,
                      processed_df: pd.DataFrame | None = None,
                      sop_data: list[str] | None = None,
                      qc_logs: pd.DataFrame | None = None) -> str:
    context_parts = []

    if isinstance(processed_df, pd.DataFrame) and "clean_text" in processed_df.columns:
        matches = processed_df[processed_df["clean_text"].str.contains(issue_text.split()[0], case=False, na=False)]
        if not matches.empty:
            context_parts.append("Past RCA Findings:")
            for _, row in matches.head(3).iterrows():
                context_parts.append(f"- {row.get('clean_text', '')[:120]}...")

    if sop_data:
        context_parts.append("Relevant SOP Snippets:")
        for snippet in sop_data[:3]:
            context_parts.append(f"- {snippet}")

    if isinstance(qc_logs, pd.DataFrame):
        numeric_cols = qc_logs.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            context_parts.append("Recent QC Metrics:")
            for col in numeric_cols[:3]:
                vals = qc_logs[col].tail(5).tolist()
                context_parts.append(f"- {col}: {vals}")

    return "\n".join(context_parts) if context_parts else "No extra context available."

# -------------------------------
# RCA Orchestration
# -------------------------------
def ai_rca_with_fallback(original_text: str,
                         clean_text: str,
                         processed_df: pd.DataFrame | None = None,
                         sop_data: list[str] | None = None,
                         qc_logs: pd.DataFrame | None = None) -> dict:
    try:
        context = build_rca_context(original_text, processed_df, sop_data, qc_logs)
        raw_result = generate_rca_with_llm(issue_text=original_text, context=context)

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
            "fishbone": convert_to_fishbone(root_causes)
        }

    except LLMRCAException:
        return huggingface_rca(original_text)
    except Exception as e:
        return {"error": f"RCA Engine Failure: {e}", "fishbone": rule_based_rca_suggestions(original_text)}

