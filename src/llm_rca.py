# ================================
# File: src/llm_rca.py
# ================================
import os
import json
import re
import time
from typing import Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None

# Try to import transformers for local HuggingFace models
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# -------------------------------
# Exceptions
# -------------------------------
class LLMRCAException(Exception):
    """Raised when the LLM RCA pipeline fails or returns invalid output."""
    pass


# -------------------------------
# API Key Helpers
# -------------------------------
def _get_api_key(provider: str = "openai") -> Optional[str]:
    key_name = "OPENAI_API_KEY" if provider == "openai" else "HUGGINGFACE_API_KEY"
    if st:
        try:
            k = st.secrets.get(key_name)
            if k:
                return k
        except Exception:
            pass
    return os.getenv(key_name)


# -------------------------------
# Prompt Template
# -------------------------------
PROMPT_TEMPLATE = """You are a senior Manufacturing Quality Engineer specialized in Root Cause Analysis (RCA).
Your task: Return a STRICT JSON object with the following structure:

{{
  "root_causes": [
    {{"cause": "<short sentence>", "category": "<Man|Machine|Method|Material|Measurement|Environment>"}}
  ],
  "five_whys": ["Why1", "Why2", "Why3", "Why4", "Why5"],
  "capa": [
    {{"type": "Corrective", "action": "<action>", "owner": "<role>", "due_in_days": <int>}},
    {{"type": "Preventive", "action": "<action>", "owner": "<role>", "due_in_days": <int>}}
  ],
  "confidence": "<low|medium|high>"
}}

Issue to analyze:
{issue_text}

Return ONLY JSON.
"""


# -------------------------------
# JSON Parsing
# -------------------------------
def _parse_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise LLMRCAException(f"Could not parse JSON from LLM response:\n{text[:400]}...")


def extract_issue_with_source(record: dict):
    """
    Extract issue text from a record dict, checking common synonyms for 'issue'.
    Returns (text, column_name).
    """
    if not isinstance(record, dict):
        return None, None

    possible_cols = [
        "issue", "issues", "problem", "problems",
        "incident", "incidents", "fault", "faults",
        "defect", "defects", "error", "errors"
    ]

    for col in possible_cols:
        if col in record and record[col]:
            return str(record[col]), col

    return None, None


# -------------------------------
# Public API
# -------------------------------
def generate_rca_with_llm(
    issue_text: str,
    context: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Generate RCA using OpenAI -> HuggingFace -> Fallback (in that order).
    """
    prompt = PROMPT_TEMPLATE.format(issue_text=issue_text + "\n\n" + context)

    # 1) Try OpenAI
    try:
        if _get_api_key("openai"):
            print("⚡ Using OpenAI for RCA")
            return _openai_rca(prompt, model=model, max_retries=max_retries, temperature=temperature)
    except Exception as e:
        print("❌ OpenAI failed:", e)

    # 2) HuggingFace
    try:
        if pipeline:
            print("⚡ Using HuggingFace pipeline for RCA")
            return _huggingface_rca(prompt)
    except Exception as e:
        print("❌ HuggingFace failed:", e)

    # 3) Fallback
    print("⚠️ Using fallback RCA")
    return _fallback_rca()


# -------------------------------
# OpenAI RCA
# -------------------------------
def _openai_rca(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 2, temperature: float = 0.0) -> Dict[str, Any]:
    api_key = _get_api_key("openai")
    if not api_key:
        raise LLMRCAException("OPENAI_API_KEY not found.")

    try:
        import openai
    except ImportError as e:
        raise LLMRCAException("openai package not installed.") from e

    openai.api_key = api_key
    attempt, last_err = 0, None

    while attempt <= max_retries:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise manufacturing RCA assistant that outputs strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=700,
            )
            content = resp["choices"][0]["message"]["content"]
            parsed = _parse_json_from_text(content)
            return parsed
        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(1.0 * attempt)

    raise LLMRCAException(f"OpenAI RCA failed after {max_retries+1} attempts. Last error: {last_err}")


# -------------------------------
# HuggingFace Local RCA
# -------------------------------
def _huggingface_rca(prompt: str) -> Dict[str, Any]:
    if not pipeline:
        raise LLMRCAException("transformers not installed.")

    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
    resp = generator(prompt, max_length=700, do_sample=True, temperature=0.2)

    if not resp or "generated_text" not in resp[0]:
        raise LLMRCAException("HuggingFace model returned empty response.")

    text = resp[0]["generated_text"]
    return _parse_json_from_text(text)


# -------------------------------
# Safe Fallback
# -------------------------------
def _fallback_rca() -> Dict[str, Any]:
    return {
        "root_causes": [
            {
                "cause": "Insufficient domain context (fallback). Review machine condition and SOP adherence.",
                "category": "Method"
            }
        ],
        "five_whys": [
            "Why was there a defect? Insufficient maintenance or SOP drift.",
            "Why insufficient? PM schedule not enforced.",
            "Why not enforced? No automated trigger or accountability.",
            "Why no trigger? SOP did not include clear checks.",
            "Why SOP gap? Review cycle missed."
        ],
        "capa": [
            {
                "type": "Corrective",
                "action": "Perform immediate equipment check and alignment verification.",
                "owner": "Maintenance",
                "due_in_days": 1
            },
            {
                "type": "Preventive",
                "action": "Add time-based PM triggers and lane-wise inspection to SOP.",
                "owner": "QA",
                "due_in_days": 14
            }
        ],
        "confidence": "low"
    }

