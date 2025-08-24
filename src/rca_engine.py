# ================================
# File: src/rca_engine.py
# ================================
import os
from typing import Dict, Any, List


# Robust import for LLM RCA
try:
# When running as a package (e.g., `python -m src.streamlit_app`)
from src.llm_rca import generate_rca_with_llm, LLMRCAException # type: ignore
except Exception:
try:
# When running from within src directory (`streamlit run src/streamlit_app.py`)
from llm_rca import generate_rca_with_llm, LLMRCAException # type: ignore
except Exception:
generate_rca_with_llm = None # fallback to HF / rule-based if not available
class LLMRCAException(Exception):
pass


try:
import requests
except Exception:
requests = None # Handle absence of requests gracefully


FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]




def generate_fishbone_skeleton() -> Dict[str, List[str]]:
return {cat: [] for cat in FISHBONE_CATEGORIES}



