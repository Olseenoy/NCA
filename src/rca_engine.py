import os
import json
from typing import Dict, Any, List
from collections import defaultdict
import requests

# Local import from your LLM module
from src.llm_rca import generate_rca_with_llm, LLMRCAException

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

def generate_fishbone_skeleton() -> Dict[str, List[str]]:
    """Create empty fishbone structure."""
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def five_whys(initial_problem: str, answers: List[str]) -> List[str]:
    """Construct 5-Whys chain."""
    chain = [initial_problem]
    for a in answers:
        if a.strip():
            chain.append(a)
    return chain

def rule_based_rca_suggestions(clean_text: str) -> Dict[str, List[str]]:
    """Generate fishbone based on keywords."""
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
    RCA via Hugging Face Inference API.
    Falls back to rule-based if API key missing or response invalid.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return {
            "error": "Hugging Face unavailable (missing API key).",
            "fishbone": rule_based_rca_suggestions(issue_text)
        }

    url = "https://api-inference.huggingface.co/models/MODEL_NAME"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": issue_text}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return {
                "error": f"Hugging Face API error: HTTP {response.status_code}",
                "fishbone": rule_based_rca_suggestions(issue_text)
            }

        try:
            data = response.json()
        except ValueError:
            return {
                "error": "Hugging Face RCA call failed: Invalid JSON response.",
                "fishbone": rule_based_rca_suggestions(issue_text)
            }

        if isinstance(data, dict) and "error" in data:
            return {
                "error": f"Hugging Face: {data['error']}",
                "fishbone": rule_based_rca_suggestions(issue_text)
            }

        # Minimal placeholder structure
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

def convert_to_fishbone(root_causes: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Convert root cause list to fishbone structure."""
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone

def ai_rca_with_fallback(original_text: str, clean_text: str) -> Dict[str, Any]:
    """
    Attempt RCA via:
        1. OpenAI LLM
        2. Hugging Face
        3. Rule-based fallback
    """
    try:
        result = generate_rca_with_llm(issue_text=original_text)
        return {
            "root_causes": result.get("root_causes", []),
            "five_whys": result.get("five_whys", []),
            "capa": result.get("capa", []),
            "fishbone": convert_to_fishbone(result.get("root_causes", []))
        }
    except LLMRCAException:
        # fallback to Hugging Face
        return huggingface_rca(original_text)
    except Exception as e:
        # final fallback to rule-based RCA
        return {
            "error": f"RCA Engine Failure: {e}",
            "fishbone": rule_based_rca_suggestions(original_text)
        }
