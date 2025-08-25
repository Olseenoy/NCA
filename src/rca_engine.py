# ================================
# File: src/rca_engine.py
# ================================
import json
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
    # Fallback placeholder
    return {
        "root_causes": [{"cause": "HF RCA cause", "category": "Method"}],
        "five_whys": ["HF Why1", "HF Why2", "HF Why3", "HF Why4", "HF Why5"],
        "capa": [
            {"type": "Corrective", "action": "HF corrective action", "owner": "QA Team", "due_in_days": 7},
            {"type": "Preventive", "action": "HF preventive action", "owner": "Maintenance", "due_in_days": 14}
        ],
        "fishbone": generate_fishbone_skeleton()
    }

def ai_rca_with_fallback(original_text: str, clean_text: str) -> dict:
    try:
        raw_result = generate_rca_with_llm(issue_text=original_text)

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

    except LLMRCAException as e:
        return huggingface_rca(original_text)
    except Exception as e:
        return {"error": f"RCA Engine Failure: {e}", "fishbone": rule_based_rca_suggestions(original_text)}
