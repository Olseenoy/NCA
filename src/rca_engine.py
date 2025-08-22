# src/rca_engine.py
from collections import defaultdict
from typing import Dict, Any

# existing categories
FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

def generate_fishbone_skeleton():
    return {cat: [] for cat in FISHBONE_CATEGORIES}


def five_whys(initial_problem: str, answers: list[str]) -> list[str]:
    """answers should be a list of user inputs for each why. Return chain."""
    chain = [initial_problem]
    for i, a in enumerate(answers, start=1):
        chain.append(a)
    return chain


def rule_based_rca_suggestions(clean_text: str) -> dict:
    """Very simple heuristics: look for keywords and map to fishbone categories"""
    keywords_map = {
        'operator': 'Man', 'training': 'Man', 'calibration': 'Measurement',
        'machine': 'Machine', 'overheat': 'Machine', 'contamination': 'Material',
        'label': 'Method', 'procedure': 'Method', 'ambient': 'Environment'
    }
    fishbone = generate_fishbone_skeleton()
    for kw, cat in keywords_map.items():
        if kw in clean_text:
            fishbone[cat].append(kw)
    return fishbone


# -------- AI wrapper that uses llm_rca and falls back to rule-based if needed --------
def ai_rca_with_fallback(issue_text: str, clean_text: str) -> Dict[str, Any]:
    """
    Try LLM-powered RCA (llm_rca.generate_rca_with_llm).
    If it fails, return a structure containing the rule-based fishbone and an error message.
    """
    try:
        # lazy import to avoid errors when not using the llm
        from llm_rca import generate_rca_with_llm
    except Exception as e:
        # cannot import LLM module; fallback to rule-based
        return {
            "error": f"LLM module not available: {e}",
            "from": "fallback_rule_based",
            "fishbone": rule_based_rca_suggestions(clean_text)
        }

    try:
        ai_result = generate_rca_with_llm(issue_text)
        # Normalize fishbone output: if AI provided root_causes as list with category, convert to map
        fishbone = generate_fishbone_skeleton()
        for rc in ai_result.get("root_causes", []):
            cause = rc.get("cause") if isinstance(rc, dict) else str(rc)
            cat = rc.get("category") if isinstance(rc, dict) else None
            if cat and cat in fishbone:
                fishbone[cat].append(cause)
            else:
                # try heuristic placement if no category
                placed = False
                lower = cause.lower()
                for kw_cat in fishbone.keys():
                    if kw_cat.lower() in lower:
                        fishbone[kw_cat].append(cause)
                        placed = True
                        break
                if not placed:
                    # fallback to 'Method'
                    fishbone['Method'].append(cause)
        ai_result["fishbone"] = fishbone
        ai_result["from"] = "llm"
        return ai_result
    except Exception as e:
        # On LLM failure, return rule-based suggestion and error
        return {
            "error": f"LLM RCA failed: {e}",
            "from": "fallback_rule_based",
            "fishbone": rule_based_rca_suggestions(clean_text)
        }
