# ==================================================================
# File: src/llm_rca.py
# ==================================================================
"""
LLM + Agent logic for RCA. Tries to use a free local LLM (Ollama via LangChain) first.
Falls back to OpenAI if key present, otherwise to the built-in lightweight HF fallback.
Drop into src/llm_rca.py
"""
import os
import json
import re
import time
from typing import Dict, Any, Optional

# Exceptions
class LLMRCAException(Exception):
    pass

# Prompt template (keeps strict JSON output)
PROMPT_TEMPLATE = """You are a senior Manufacturing Quality Engineer specialized in Root Cause Analysis (RCA).
Given: structured_issue, raw_text, and context (SOPs / maintenance / QC trends).
Return EXACTLY one JSON object matching this schema:
{
  "root_causes": [{"cause": "<short>", "category": "<Man|Machine|Method|Material|Measurement|Environment>"}],
  "five_whys": ["Why1","Why2","Why3","Why4","Why5"],
  "capa": [{"type":"Corrective","action":"...","owner":"...","due_in_days":int},{"type":"Preventive","action":"...","owner":"...","due_in_days":int}],
  "confidence": "<low|medium|high>",
  "analysis": "<short summary>"
}

Structured issue:
{structured_issue}

Raw text:
{raw_text}

Context:
{context_block}
"""

# Robust parser
def _parse_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        candidate = re.sub(r",\s*}\s*$", "}", candidate)
        candidate = re.sub(r",\s*]\s*$", "]", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise LLMRCAException("Could not parse JSON from LLM response.")

# Primary entrypoint
def generate_rca_with_llm(issue_text: str, context: str = "", max_retries: int = 2, temperature: float = 0.0) -> Dict[str, Any]:
    # Heuristics
    structured_issue = "(not provided)"
    raw_text = issue_text or "(not provided)"

    if "Parsed context" in issue_text or '{' in issue_text:
        structured_issue = issue_text
        raw_text = "(not provided)"

    prompt = PROMPT_TEMPLATE.format(structured_issue=structured_issue, raw_text=raw_text, context_block=context or "No context provided.")

    # Try local Ollama via LangChain first (free local option)
    try:
        from langchain.llms import Ollama
        llm = Ollama(model="mistral")
        resp = llm(prompt)
        parsed = _parse_json_from_text(resp)
        return parsed
    except Exception:
        # proceed to next option
        pass

    # Try OpenAI if API key present
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            attempt = 0
            last_err = None
            while attempt <= max_retries:
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "You are a precise manufacturing RCA assistant."}, {"role": "user", "content": prompt}],
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
            raise LLMRCAException(f"OpenAI failed: {last_err}")
        except Exception:
            pass

    # Final fallback: lightweight HF or in-code fallback
    return _huggingface_rca(raw_text)

# Simple HuggingFace / Offline fallback (keeps conservative template)
def _huggingface_rca(issue_text: str) -> Dict[str, Any]:
    return {
        "root_causes": [
            {"cause": "Potential equipment wear or misalignment.", "category": "Machine"},
            {"cause": "SOP adherence gaps.", "category": "Method"},
        ],
        "five_whys": [
            "Why? Equipment condition degraded.",
            "Why? PM interval not followed.",
            "Why? No runtime-based triggers.",
            "Why? SOP lacks checks.",
            "Why? Review cycle missed."
        ],
        "capa": [
            {"type": "Corrective", "action": "Inspect and realign tooling; verify torque and replace worn parts.", "owner": "Maintenance", "due_in_days": 1},
            {"type": "Preventive", "action": "Add runtime-based PM triggers and update SOP with lane-wise checks.", "owner": "QA/Production", "due_in_days": 21}
        ],
        "confidence": "low",
        "analysis": "Fallback RCA: needs human validation"
        }
