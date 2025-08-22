# src/rca_engine.py
import os
import json
from collections import defaultdict
import openai

FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

def generate_fishbone_skeleton():
    return {cat: [] for cat in FISHBONE_CATEGORIES}

def five_whys(initial_problem: str, answers: list[str]) -> list[str]:
    chain = [initial_problem]
    for a in answers:
        if a.strip():
            chain.append(a)
    return chain

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

def ai_rca_with_fallback(original_text: str, clean_text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not found in Streamlit secrets or environment variables."}

    openai.api_key = api_key
    prompt = f"""
Perform a root cause analysis (RCA) on this issue:
---
{original_text}
---
Output JSON with keys:
- "root_causes": list of objects with "category" (Man, Machine, Method, Material, Measurement, Environment) and "cause"
- "five_whys": list of 5 strings
- "capa": list of objects with "type" (Corrective or Preventive), "action", "owner", "due_in_days"
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an RCA assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=400
        )
        raw_text = response.choices[0].message["content"]
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "error": "AI output not valid JSON. Falling back to rule-based RCA.",
            "fishbone": rule_based_rca_suggestions(clean_text)
        }
    except Exception as e:
        return {
            "error": f"LLM RCA failed: {e}",
            "fishbone": rule_based_rca_suggestions(clean_text)
        }

    # Ensure all keys exist
    return {
        "root_causes": result.get("root_causes", []),
        "five_whys": result.get("five_whys", []),
        "capa": result.get("capa", []),
        "fishbone": convert_to_fishbone(result.get("root_causes", []))
    }

def convert_to_fishbone(root_causes):
    fishbone = generate_fishbone_skeleton()
    for rc in root_causes:
        if isinstance(rc, dict):
            cat = rc.get("category", "Method")
            fishbone.setdefault(cat, []).append(rc.get("cause", ""))
    return fishbone
