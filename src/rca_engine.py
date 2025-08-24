# ================================
# File: src/rca_engine.py
# ================================
import os
from typing import Dict, Any, List

# Robust import for LLM RCA
try:
    # When running as a package (e.g., `python -m src.streamlit_app`)
    from src.llm_rca import generate_rca_with_llm, LLMRCAException  # type: ignore
except Exception:
    try:
        # When running from within src directory (`streamlit run src/streamlit_app.py`)
        from llm_rca import generate_rca_with_llm, LLMRCAException  # type: ignore
    except Exception:
        generate_rca_with_llm = None  # fallback to HF / rule-based if not available
        class LLMRCAException(Exception):
            pass

try:
    import requests
except Exception:
    requests = None  # Handle absence of requests gracefully

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]


def generate_fishbone_skeleton() -> Dict[str, List[str]]:
    return {cat: [] for cat in FISHBONE_CATEGORIES}


def five_whys(initial_problem: str, answers: List[str]) -> List[str]:
    chain = [initial_problem]
    for a in answers:
        if str(a).strip():
            chain.append(str(a))
    return chain


def rule_based_rca_suggestions(clean_text: str) -> Dict[str, List[str]]:
    keywords_map = {
        'operator': 'Man', 'training': 'Man', 'calibration': 'Measurement',
        'machine': 'Machine', 'overheat': 'Machine', 'contamination': 'Material',
        'label': 'Method', 'procedure': 'Method', 'ambient': 'Environment'
    }
    fishbone = generate_fishbone_skeleton()
    text = (clean_text or "").lower()
    for kw, cat in keywords_map.items():
        if kw.lower() in text:
            fishbone[cat].append(kw)
    return fishbone


def huggingface_rca(issue_text: str) -> Dict[str, Any]:
    """
    Uses Hugging Face Inference API for RCA if OpenAI fails.
    Replace 'MODEL_NAME' with an appropriate model.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key or not requests:
        return {"error": "Hugging Face unavailable (missing API key or requests).", "fishbone": rule_based_rca_suggestions(issue_text)}

    url = "https://api-inference.huggingface.co/models/MODEL_NAME"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": issue_text}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            return {"error": f"Hugging Face: {data['error']}", "fishbone": rule_based_rca_suggestions(issue_text)}
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
        return {"error": f"Hugging Face RCA call failed: {e}", "fishbone": rule_based_rca_suggestions(issue_text)}


def convert_to_fishbone(root_causes: Any) -> Dict[str, List[str]]:
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes or []:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method") or "Method"
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone


def ai_rca_with_fallback(original_text: str, clean_text: str) -> Dict[str, Any]:
    """
    Attempt RCA via OpenAI LLM -> Hugging Face -> Rule-based
    """
    # 1) Try OpenAI-based LLM if available
    if callable(generate_rca_with_llm):
        try:
            result = generate_rca_with_llm(issue_text=str(original_text))
            return {
                "root_causes": result.get("root_causes", []),
                "five_whys": result.get("five_whys", []),
                "capa": result.get("capa", []),
                "fishbone": convert_to_fishbone(result.get("root_causes", [])),
            }
        except LLMRCAException:
            # fall through to next level
            pass
        except Exception:
            # Any unexpected error: continue to fallback
            pass

    # 2) Try Hugging Face (if configured)
    hf = huggingface_rca(str(original_text))
    if not hf.get("error") or hf.get("fishbone"):
        return hf

    # 3) Rule-based fallback
    return {"fishbone": rule_based_rca_suggestions(str(clean_text))}
