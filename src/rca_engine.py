# ================================
# File: src/rca_engine.py
# ================================
import json
import re
from typing import Dict, Any
from src.llm_rca import generate_rca_with_llm, LLMRCAException

# Predefined Fishbone Categories
FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

def generate_fishbone_skeleton() -> Dict[str, list]:
    """Creates an empty fishbone structure."""
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def convert_to_fishbone(root_causes: list) -> Dict[str, list]:
    """Converts root causes to fishbone diagram data."""
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            cause = rc.get("cause", "")
            if cause:
                fishbone.setdefault(cat, []).append(cause)
    return fishbone

def sanitize_llm_output(output: str) -> str:
    """
    Cleans and extracts valid JSON from LLM response.
    Handles cases where extra text, markdown, or newlines are present.
    """
    cleaned = output.strip()

    # Remove Markdown code blocks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)  # remove ```json or ```
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # Extract JSON portion (first { to last })
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    return cleaned

def rule_based_rca_suggestions(clean_text: str) -> Dict[str, list]:
    """
    Generates a basic RCA suggestion using keyword mapping.
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
    Lightweight fallback RCA output for when LLM fails.
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
            {"type": "Corrective", "action": "Assign RCA to QA team", "owner": "QA Team", "due_in_days": 3},
            {"type": "Preventive", "action": "Establish backup RCA system", "owner": "Maintenance", "due_in_days": 14}
        ],
        "fishbone": generate_fishbone_skeleton(),
        "confidence": "low"
    }

def ai_rca_with_fallback(original_text: str, clean_text: str, mode: str = "local") -> Dict[str, Any]:
    """
    Primary RCA inference pipeline with sanitization, error handling & fallback.
    """
    try:
        raw_result = generate_rca_with_llm(issue_text=original_text, mode=mode)

        # Ensure we parse JSON correctly
        if isinstance(raw_result, str):
            try:
                sanitized = sanitize_llm_output(raw_result)
                result = json.loads(sanitized)
            except json.JSONDecodeError as e:
                return {
                    "error": f"LLM JSON decode error: {e}",
                    "fishbone": rule_based_rca_suggestions(clean_text),
                    "confidence": "low"
                }
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            return {
                "error": "LLM returned unexpected type.",
                "fishbone": rule_based_rca_suggestions(clean_text),
                "confidence": "low"
            }

        # Extract RCA components
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
        return huggingface_rca(original_text)
    except Exception as e:
        return {
            "error": f"RCA Engine Failure: {e}",
            "fishbone": rule_based_rca_suggestions(clean_text),
            "confidence": "low"
        }
