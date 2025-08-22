# src/llm_rca.py
import os
import json
import time
import re
from typing import Dict, Any

# Attempt to import streamlit for secrets retrieval; keep optional to allow non-streamlit usage.
try:
    import streamlit as st
except Exception:
    st = None

class LLMRCAException(Exception):
    pass

def _get_api_key() -> str | None:
    """
    Check Streamlit secrets first (if available), otherwise environment variable.
    """
    if st:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
            if key:
                return key
        except Exception:
            pass
    return os.getenv("OPENAI_API_KEY")


PROMPT_TEMPLATE = '''You are an expert Quality Assurance engineer specializing in manufacturing Root Cause Analysis (RCA).
Given the non-conformance report below, produce a JSON-only response (no extra text) with this exact schema:

{
  "root_causes": [
    {"cause": "<short sentence>", "category": "<Man|Machine|Method|Material|Measurement|Environment>"}
  ],
  "five_whys": ["Why1 answer", "Why2 answer", "Why3 answer", "Why4 answer", "Why5 answer"],
  "capa": [
    {"type": "Corrective"|"Preventive", "action": "<concrete action>", "owner": "<role or team>", "due_in_days": <int>}
  ],
  "confidence": "<low|medium|high>"
}

Be concise. Give 3-5 root_causes (as array), exactly 5 five_whys entries, 2 CAPA entries (one corrective, one preventive) with realistic due_in_days.
Issue:
\"\"\"{issue_text}\"\"\"
Respond only with valid JSON that matches the schema above.
'''

def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON from LLM text. If LLM emits extra commentary, attempt to extract the JSON block.
    """
    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract {...} block
    m = re.search(r'(\{.*\})', text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract last JSON-like substring (robust fallback)
    matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text, re.S)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise LLMRCAException("Could not parse JSON from LLM response")

def generate_rca_with_llm(issue_text: str, model: str = "gpt-4o-mini", max_retries: int = 2, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Call OpenAI (ChatCompletion) to generate structured RCA output.
    - Uses Streamlit secrets or OPENAI_API_KEY env var.
    - Retries on transient failures.
    """
    api_key = _get_api_key()
    if not api_key:
        raise LLMRCAException("OPENAI_API_KEY not found in Streamlit secrets or environment variables.")

    # Lazy import openai to avoid forcing dependency if module not used
    try:
        import openai
    except ImportError as e:
        raise LLMRCAException("openai package not installed. Install with `pip install openai`.") from e

    openai.api_key = api_key
    prompt = PROMPT_TEMPLATE.format(issue_text=issue_text)

    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a manufacturing quality expert."},
                          {"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=temperature,
            )
            content = resp["choices"][0]["message"]["content"]
            parsed = _parse_json_from_text(content)
            # Basic schema checks and normalization
            if "root_causes" not in parsed or "five_whys" not in parsed or "capa" not in parsed:
                raise LLMRCAException("LLM returned JSON but missing required keys.")
            # Ensure five_whys length = 5
            if not isinstance(parsed.get("five_whys"), list) or len(parsed.get("five_whys")) != 5:
                # attempt to coerce: split lines, take first 5
                f = parsed.get("five_whys")
                if isinstance(f, str):
                    f_list = [s.strip() for s in re.split(r'[\n\r]+', f) if s.strip()][:5]
                    parsed["five_whys"] = f_list + [""]*(5-len(f_list))
                else:
                    parsed["five_whys"] = (f or [])[:5] + [""]*(5 - len((f or [])))
            return parsed
        except Exception as e:
            last_err = e
            attempt += 1
            wait = 1.0 * attempt
            time.sleep(wait)
            continue

    raise LLMRCAException(f"LLM calls failed after {max_retries+1} attempts. Last error: {last_err}")
