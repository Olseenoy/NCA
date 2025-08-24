import os
import json
from typing import Dict, Any
from collections import defaultdict
import requests
from src.llm_rca import generate_rca_with_llm, LLMRCAException

# Standard fishbone categories
FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

def generate_fishbone_skeleton() -> dict:
    """Create an empty fishbone structure with all categories."""
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def five_whys(initial_problem: str, answers: list[str]) -> list[str]:
    """Generate a 5-Whys chain starting from initial problem and answers."""
    chain = [initial_problem]
    for a in answers:
        if a.strip():
            chain.append(a)
    return chain

def rule_based_rca_suggestions(clean_text: str) -> dict:
    """
    Fallback RCA logic using simple keyword matching.
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
    Use Hugging Face Inference API for RCA if OpenAI LLM fails.
    Replace 'MODEL_NAME' with a valid Hugging Face model.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return {
            "error": "Hugging Face unavailable (missing API key or requests).",
            "fishbone": rule_based_rca_suggestions(issue_text)
        }

    url = "https://api-inference.huggingface.co/models/MODEL_NAME"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": issue_text}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return {
                "error": f"Hugging Face call failed: {response.status_code}",
                "fishbone": rule_based_rca_suggestions(issue_text)
            }

        data = response.json()
        if not isinstance(data, dict):
            return {
                "error": "Unexpected Hugging Face response format.",
                "fishbone": rule_based_rca_suggestions(issue_text)
            }

        # Minimal placeholder extraction
        return {
            "root_causes": [{"cause": "HF RCA cause", "category": "Method"}],
            "five_whys": ["HF Why1", "HF Why2", "HF Why3", "HF Why4", "HF Why5"],
            "capa": [
                {"type": "Corrective", "action": "HF corrective action", "owner": "QA Team", "due_in_days": 7},
                {"type": "Preventive", "action": "HF preventive action", "owner": "Maintenance", "due_in_days": 14}
            ],
            "fishbone": generate_fishbone_skeleton()
        }

    except Exception as e:
        return {
            "error": f"Hugging Face RCA call failed: {e}",
            "fishbone": rule_based_rca_suggestions(issue_text)
        }

def ai_rca_with_fallback(original_text: str, clean_text: str) -> Dict[str, Any]:
    """
    Attempt RCA via:
        1. OpenAI LLM
        2. Hugging Face
        3. Rule-based fallback
    """
    try:
        raw_result = generate_rca_with_llm(issue_text=original_text)

        # Ensure result is dict
        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                return {
                    "error": "LLM returned invalid JSON.",
                    "fishbone": rule_based_rca_suggestions(original_text)
                }
        elif isinstance(raw_result, dict):
            result = raw_result
        else:
            return {
                "error": "LLM returned unexpected type.",
                "fishbone": rule_based_rca_suggestions(original_text)
            }

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
        return {
            "error": f"RCA Engine Failure: {e}",
            "fishbone": rule_based_rca_suggestions(original_text)
        }

def convert_to_fishbone(root_causes):
    """Convert root causes to fishbone structure."""
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone
