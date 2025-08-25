# ================================
# File: src/rca_engine.py
# ================================
import json
from typing import Dict, Any, List, Union
from src.llm_rca import generate_rca_with_llm, LLMRCAException

# Standard fishbone categories
FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]


def generate_fishbone_skeleton() -> Dict[str, List[str]]:
    """
    Creates an empty fishbone diagram structure with categories as keys.
    """
    return {cat: [] for cat in FISHBONE_CATEGORIES}


def convert_to_fishbone(root_causes: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Converts root causes from LLM RCA output into a fishbone diagram structure.
    """
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone


def rule_based_rca_suggestions(clean_text: str) -> Dict[str, List[str]]:
    """
    Provides fallback RCA suggestions using simple keyword mapping.
    """
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


def huggingface_rca(issue_text: str) -> Dict[str, Any]:
    """
    Lightweight fallback RCA output for when LLM and local pipeline both fail.
    """
    return {
        "root_causes": [{"cause": "Manual RCA required", "category": "Method"}],
        "five_whys": [
            "Why1: Investigation required",
            "Why2: Pending",
            "Why3: Pending",
            "Why4: Pending",
            "Why5: Pending"
        ],
        "capa": [
            {
                "type": "Corrective",
                "action": "Assign RCA to QA team",
                "owner": "QA Team",
                "due_in_days": 3
            },
            {
                "type": "Preventive",
                "action": "Establish backup RCA system",
                "owner": "Maintenance",
                "due_in_days": 14
            }
        ],
        "fishbone": generate_fishbone_skeleton(),
        "confidence": "low"
    }


def ai_rca_with_fallback(original_text: str,
                         clean_text: str,
                         mode: str = "local") -> Dict[str, Any]:
    """
    Primary RCA inference pipeline with multiple fallback layers.
    - Tries LLM-based RCA first.
    - Falls back to HuggingFace or rule-based approach on failure.
    """
    try:
        raw_result: Union[str, Dict[str, Any]] = generate_rca_with_llm(issue_text=original_text, mode=mode)

        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                return {
                    "error": "LLM returned invalid JSON.",
                    "fishbone": rule_based_rca_suggestions(clean_text),
                    "confidence": "low"
                }
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            return {
                "error": "LLM returned unexpected response type.",
                "fishbone": rule_based_rca_suggestions(clean_text),
                "confidence": "low"
            }

        # Extract RCA components with safe defaults
        root_causes = result.get("root_causes", [])
        five_whys_chain = result.get("five_whys", [])
        capa_actions = result.get("capa", [])
        confidence = result.get("confidence", "medium")

        return {
            "root_causes": root_causes,
            "five_whys": five_whys_chain,
            "capa": capa_actions,
            "fishbone": convert_to_fishbone(root_causes),
            "confidence": confidence
        }

    except LLMRCAException:
        # If LLM RCA fails entirely, return HuggingFace fallback response
        return huggingface_rca(original_text)

    except Exception as e:
        # Final fallback: rule-based + error note
        return {
            "error": f"RCA Engine Failure: {e}",
            "fishbone": rule_based_rca_suggestions(clean_text),
            "confidence": "low"
        }
