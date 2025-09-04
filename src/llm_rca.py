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
# API Key Helpers
# -------------------------------
def _get_api_key(provider: str = "openai") -> Optional[str]:
    """
    Get API key from Streamlit secrets (preferred) or environment variables.
    """
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
# Prompt (log-friendly + context-aware)
# -------------------------------
PROMPT_TEMPLATE = """You are a senior Manufacturing Quality Engineer specialized in Root Cause Analysis (RCA).
You are given:
1) A structured interpretation of a production issue (parsed fields like date, shift, machine, lanes, time range, defect),
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
- Be concrete and operational; avoid generic statements.
- If context mentions maintenance checks (e.g., torque, blade wear, alignment), consider them where relevant.

=== STRUCTURED_ISSUE ===
{structured_issue}

=== RAW_TEXT ===
{raw_text}

=== CONTEXT ===
{context_block}

Return ONLY the JSON object. No extra text.
"""


# -------------------------------
# JSON Parsing (robust)
# -------------------------------
def _parse_json_from_text(text: str) -> dict:
    """
    Robustly parse JSON from LLM response.
    Strips any accidental pre/post text, tolerates trailing commas.
    """
    text = (text or "").strip()

    # Quick path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract between first { and last }
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        # Remove trailing commas before } or ]
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
    """
    Run RCA using OpenAI Chat Completions with a context-aware, log-friendly prompt.
    - issue_text: can be either raw issue text OR a combined/structured block. (Compatible with the orchestrator.)
    - context: optional extra knowledge (SOP snippets, QC trends, past RCAs).
    """
    # For backward compatibility with callers that already combined context into issue_text:
    # If the caller passes a full composed block, we detect and place it into structured_issue/raw_text slots.
    structured_issue = ""
    raw_text = issue_text

    # Heuristic: if issue_text already contains "Parsed context:" or looks like JSON, treat it as structured.
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
        # Fallback path
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

    # Legacy client compatibility
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

            # Validate essential keys
            for k in ("root_causes", "five_whys", "capa", "confidence"):
                if k not in parsed:
                    raise LLMRCAException(f"Incomplete JSON from OpenAI RCA: missing '{k}'")
            # Minimal shape checks
            if not isinstance(parsed.get("root_causes"), list) or not isinstance(parsed.get("capa"), list):
                raise LLMRCAException("Invalid JSON structure for 'root_causes' or 'capa'.")

            return parsed

        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(1.0 * attempt)

    raise LLMRCAException(f"OpenAI RCA failed after {max_retries+1} attempts. Last error: {last_err}")


# -------------------------------
# Hugging Face Fallback (placeholder)
# -------------------------------
def _huggingface_rca(issue_text: str) -> Dict[str, Any]:
    """
    VERY lightweight fallback that calls HF Inference API if configured,
    otherwise returns a conservative, generic RCA result.
    """
    api_key = _get_api_key("huggingface")
    if not api_key:
        # Offline fallback
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
                {"type": "Corrective", "action": "Perform immediate equipment check and alignment verification.", "owner": "Maintenance", "due_in_days": 1},
                {"type": "Preventive", "action": "Add time-based PM triggers and lane-wise inspection to SOP.", "owner": "QA", "due_in_days": 14}
            ],
            "confidence": "low"
        }

    # If user has an HF model, call it here (placeholder)
    url = "https://api-inference.huggingface.co/models/MODEL_NAME"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": issue_text or "manufacturing rca"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        _ = r.json()  # Not used: each HF model differs; we return a safe template
    except Exception:
        pass

    # Generic minimal fallback JSON
    return {
        "root_causes": [
            {"cause": "Potential equipment wear or misalignment (fallback).", "category": "Machine"},
            {"cause": "SOP checks not consistently executed.", "category": "Method"}
        ],
        "five_whys": [
            "Defect occurred because perforation quality degraded.",
            "Quality degraded due to blade wear or torque drift.",
            "Wear/drift persisted because PM interval exceeded.",
            "PM interval exceeded because no runtime-based triggers.",
            "No triggers because SOP is time-based only."
        ],
        "capa": [
            {"type": "Corrective", "action": "Replace/realign critical tooling; verify torque settings.", "owner": "Maintenance", "due_in_days": 1},
            {"type": "Preventive", "action": "Introduce runtime-based PM triggers and lane-wise inspection checks.", "owner": "Production/QA", "due_in_days": 21}
        ],
        "confidence": "low"
    }
