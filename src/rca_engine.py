# src/rca_engine.py
from collections import defaultdict

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
