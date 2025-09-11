# ================================
# File: src/llm_rca.py
# ================================
import os
import json
import re
import time
import requests
from typing import Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None


# -------------------------------
# Exceptions
# -------------------------------
class LLMRCAException(Exception):
    """Raised when the LLM RCA pipeline fails or returns invalid output."""
    pass


# -------------------------------
# Issue Text Extractor (synonyms-aware)
# -------------------------------
ISSUE_COLS = [
    "issue_description", "issue", "issues",
    "problem", "defect", "failure",
    "incident", "observation", "error", "complaint"
]

def extract_issue_with_source(record: dict) -> (str, str):
    """
    Extract issue/problem text and the column name it came from.
    Falls back to combined_text or clean_text.
    """
    for col in ISSUE_COLS:
        if col in record and record[col]:
            return str(record[col]), col
    if "combined_text" in record and record["combined_text"]:
        return str(record["combined_text"]), "combined_text"
    if "clean_text" in record and record["clean_text"]:
        return str(record["clean_text"]), "clean_text"
    return "", "unknown"


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
# Prompt
# -------------------------------
PROMPT_TEMPLATE = """You are a senior Manufacturing Quality Engineer specialized in Root Cause Analysis (RCA).
You are given:
1) A structured interpretation of a production issue,
2) The original raw text (as logged on the shop floor),
3) Optional context: past RCAs, SOP snippets, QC trends.

Your task: Return a STRICT JSON object (no prose) following EXACTLY this schema:

{{
  "root_causes": [
    {{"cause": "<short sentence>", "category": "<Man|Machine|Method|Material|Measurement|Environment>"}}
  ],
  "five_whys": ["Why1 answer", "Why2 answer", "Why3 answer", "Why4 answer", "Why5 answer"],
  "capa": [
    {{"type": "Corrective", "action": "<concrete action>", "owner": "<role or team>", "due_in_days": <int>}},
    {{"type": "Preventive", "action": "<concrete action>", "owner": "<role or team>", "due_in_days": <int>}}
  ],
  "confidence": "<low|medium|high>"
}}

Rules:
- 3–5 items in "root_causes".
- Exactly 5 entries in "five_whys".
- Exactly 2 CAPA entries: one Corrective and one Preventive.
- Categories limited to: Man, Machine, Method, Material, Measurement, Environment.
- Be concrete and operational.

=== STRUCTURED_ISSUE ===
{structured_issue}

=== RAW_TEXT ===
{raw_text}

=== CONTEXT ===
{context_block}

Return ONLY the JSON object.
"""


# -------------------------------
# JSON Parsing (robust)
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


# -------------------------------
# Public API
# -------------------------------
def generate_rca_with_llm(
    issue_text: str,
    context: str = "",
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    structured_issue = ""
    raw_text = issue_text

    if "Parsed context" in issue_text or re.search(r'"\s*defect"\s*:', issue_text):
        structured_issue = issue_text
        raw_text = ""

    prompt = PROMPT_TEMPLATE.format(
        structured_issue=structured_issue or "(not provided)",
        raw_text=raw_text or "(not provided)",
        context_block=context or "No additional context provided."
    )

    try:
        return _openai_rca(prompt, model=model, max_retries=max_retries, temperature=temperature)
    except LLMRCAException:
        return _huggingface_rca(raw_text or structured_issue)


# -------------------------------
# OpenAI Client
# -------------------------------
def _openai_rca(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    api_key = _get_api_key("openai")
    if not api_key:
        raise LLMRCAException("OPENAI_API_KEY not found.")

    try:
        import openai
    except ImportError as e:
        raise LLMRCAException("openai package not installed.") from e

    openai.api_key = api_key
    attempt = 0
    last_err = None

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

            for k in ("root_causes", "five_whys", "capa", "confidence"):
                if k not in parsed:
                    raise LLMRCAException(f"Incomplete JSON from OpenAI RCA: missing '{k}'")
            return parsed

        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(1.0 * attempt)

    raise LLMRCAException(f"OpenAI RCA failed after {max_retries+1} attempts. Last error: {last_err}")


# -------------------------------
# Hugging Face Fallback
# -------------------------------
def _huggingface_rca(issue_text: str) -> Dict[str, Any]:
    return {
        "root_causes": [
            {"cause": "Insufficient domain context (fallback). Review machine condition and SOP adherence.", "category": "Method"}
        ],
        "five_whys": [
            "Why was there a defect? Insufficient maintenance or SOP drift.",
            "Why insufficient? PM schedule not enforced.",
            "Why not enforced? No automated trigger or accountability.",
            "Why no trigger? SOP did not include clear checks.",
            "Why SOP gap? Review cycle missed."
        ],
        "capa": [
            {"type": "Corrective", "action": "Perform immediate equipment check.", "owner": "Maintenance", "due_in_days": 1},
            {"type": "Preventive", "action": "Add time-based PM triggers.", "owner": "QA", "due_in_days": 14}
        ],
        "confidence": "low"
    }
