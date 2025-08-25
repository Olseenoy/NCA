# ================================
# File: src/llm_rca.py
# ================================
import os
import json
import re
import time
import requests
from typing import Dict, Any

try:
    import streamlit as st
except Exception:
    st = None

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from crewai import Agent, Task, Crew, Process

# Custom exception for LLM failures
class LLMRCAException(Exception):
    pass

def generate_rca_with_llm(issue_text: str, mode: str = "local") -> Dict[str, Any]:
    """
    Generates Root Cause Analysis (RCA) using an AI model.
    Returns valid JSON or raises LLMRCAException on failure.
    """

    # Example local inference simulation (replace with LangChain / Hugging Face pipeline)
    if mode == "local":
        return _mock_local_llm(issue_text)

    # Placeholder for other modes (Hugging Face API, CrewAI, etc.)
    elif mode == "huggingface":
        return _mock_huggingface_llm(issue_text)

    else:
        raise LLMRCAException(f"Unsupported LLM mode: {mode}")

def _mock_local_llm(issue_text: str) -> Dict[str, Any]:
    """
    Simulates local model response in JSON.
    Replace this with actual LangChain or Hugging Face pipeline call.
    """
    # Minimal rule-based generation for now
    return {
        "root_causes": [
            {"cause": f"Potential process issue with: {issue_text[:40]}...", "category": "Method"},
            {"cause": "Possible human error", "category": "Man"}
        ],
        "five_whys": [
            f"Why1: Why did {issue_text[:30]} occur?",
            "Why2: Root cause still under analysis",
            "Why3: Contributing factors may include training gaps",
            "Why4: Lack of standard operating procedure review",
            "Why5: Incomplete CAPA implementation"
        ],
        "capa": [
            {"type": "Corrective", "action": "Perform detailed RCA on equipment and training gaps", "owner": "QA Team", "due_in_days": 7},
            {"type": "Preventive", "action": "Review SOPs and retrain operators", "owner": "Operations", "due_in_days": 14}
        ],
        "confidence": "medium"
    }

def _mock_huggingface_llm(issue_text: str) -> Dict[str, Any]:
    """
    Placeholder for Hugging Face pipeline.
    Should format response as valid JSON.
    """
    return {
        "root_causes": [
            {"cause": f"Hugging Face analysis: {issue_text[:40]}...", "category": "Machine"}
        ],
        "five_whys": [
            f"Why1: Investigating {issue_text[:30]}...",
            "Why2: Pending deeper model analysis",
            "Why3: Pending",
            "Why4: Pending",
            "Why5: Pending"
        ],
        "capa": [
            {"type": "Corrective", "action": "Assign RCA review to data science team", "owner": "Data Science", "due_in_days": 10}
        ],
        "confidence": "medium"
    }
Key Fixes
Always returns a Python dictionary, not a raw string, so json.loads() is only a fallback for text-based models.

Added _mock_local_llm() and _mock_huggingface_llm() to simulate responses while integrating real LLM pipelines later.

Clear LLMRCAException for unsupported modes.

Would you like me to:

Integrate Hugging Face local pipeline (transformers or text-generation-inference) now?

Or keep mock mode until your LangChain + CrewAI + HF pipeline is ready?

Which do you prefer for your first working version? Mock or real Hugging Face pipeline?







You said:
REWRITE FULL LLM_RCA


ChatGPT said:

Thought for 22s
Here’s a robust, drop-in replacement for src/llm_rca.py that guarantees a valid dict is always returned, adds solid sanitization/validation, supports multiple modes (local, openai, api_hf, and an optional local_llm), and gracefully falls back to a deterministic rule-based RCA if anything goes wrong.

You don’t need to change other files for this to work with your current rca_engine.py.

src/llm_rca.py
python
Copy
Edit
# ================================
# File: src/llm_rca.py
# ================================
import os
import json
import re
import time
from typing import Dict, Any, List, Optional

try:
    import streamlit as st
except Exception:
    st = None

# Optional dependencies (guarded)
try:
    import requests
except Exception:
    requests = None

try:
    from transformers import pipeline as hf_pipeline  # optional
except Exception:
    hf_pipeline = None

try:
    # Optional LangChain/CrewAI wrappers (used only if available AND mode="local_llm")
    from langchain.prompts import PromptTemplate
    from langchain.llms import HuggingFacePipeline
    from crewai import Agent, Task, Crew, Process
except Exception:
    PromptTemplate = None
    HuggingFacePipeline = None
    Agent = Task = Crew = Process = None


# -----------------------------
# Exceptions
# -----------------------------
class LLMRCAException(Exception):
    """Raised when an LLM-driven RCA flow fails in a way that should trigger fallback."""
    pass


# -----------------------------
# Constants & Prompt
# -----------------------------
FISHBONE_CATEGORIES = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]

PROMPT_TEMPLATE = (
    'You are an expert Quality Assurance engineer specializing in manufacturing Root Cause Analysis (RCA).\n'
    'Given the non-conformance report below, produce a JSON-only response with this exact schema:\n\n'
    '{\n'
    '  "root_causes": [\n'
    '    {"cause": "<short sentence>", "category": "<Man|Machine|Method|Material|Measurement|Environment>"}\n'
    '  ],\n'
    '  "five_whys": ["Why1 answer", "Why2 answer", "Why3 answer", "Why4 answer", "Why5 answer"],\n'
    '  "capa": [\n'
    '    {"type": "Corrective"|"Preventive", "action": "<concrete action>", "owner": "<role or team>", "due_in_days": <int>}\n'
    '  ],\n'
    '  "confidence": "<low|medium|high>"\n'
    '}\n\n'
    "Constraints:\n"
    "- Return 3 to 5 root_causes.\n"
    "- Exactly 5 five_whys entries.\n"
    "- Exactly 2 CAPA entries (1 Corrective, 1 Preventive).\n"
    "- Respond ONLY with valid JSON.\n\n"
    "Issue:\n\"\"\"{issue_text}\"\"\"\n"
)


# -----------------------------
# Secrets / API keys
# -----------------------------
def _get_api_key(provider: str) -> Optional[str]:
    """Gets keys from Streamlit secrets (if present) or env vars."""
    key_name = "OPENAI_API_KEY" if provider == "openai" else "HUGGINGFACE_API_KEY"
    if st:
        try:
            val = st.secrets.get(key_name)
            if val:
                return val
        except Exception:
            pass
    return os.getenv(key_name)


# -----------------------------
# Sanitization, Parsing, Validation
# -----------------------------
def _sanitize_llm_text_to_json(text: str) -> str:
    """Remove code fences and extract the first {...} JSON block."""
    if not text:
        raise LLMRCAException("Empty LLM output.")
    cleaned = text.strip()

    # Remove Markdown code fences like ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)  # leading fence w/ optional lang
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Extract JSON between the first { and last }
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise LLMRCAException(f"No JSON object found in LLM output (starts with: {cleaned[:80]!r})")
    return m.group(0)


def _validate_and_coerce_schema(data: Dict[str, Any], issue_text: str) -> Dict[str, Any]:
    """
    Ensure required keys exist and have valid shapes.
    Coerce to spec if needed so the caller always gets a valid structure.
    """
    # Required sections
    root_causes: List[Dict[str, str]] = data.get("root_causes") or []
    five_whys: List[str] = data.get("five_whys") or []
    capa: List[Dict[str, Any]] = data.get("capa") or []
    confidence: str = (data.get("confidence") or "medium").lower()

    # Normalize confidence
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    # --- Root causes: 3-5, each with cause + category (valid set)
    # If invalid or too few, augment using deterministic extraction
    root_causes = [
        rc for rc in root_causes
        if isinstance(rc, dict) and rc.get("cause") and rc.get("category") in FISHBONE_CATEGORIES
    ]

    if len(root_causes) < 3:
        root_causes = _augment_root_causes(issue_text, root_causes)
    if len(root_causes) > 5:
        root_causes = root_causes[:5]

    # --- Five Whys: exactly 5
    five_whys = [str(x) for x in five_whys if isinstance(x, (str, int))]
    if len(five_whys) < 5:
        five_whys += [f"Why{idx}: Pending analysis" for idx in range(len(five_whys) + 1, 6)]
    elif len(five_whys) > 5:
        five_whys = five_whys[:5]

    # --- CAPA: exactly 2 (1 Corrective, 1 Preventive)
    capa = [c for c in capa if isinstance(c, dict)]
    _ensure_two_capa_entries(capa)

    return {
        "root_causes": root_causes,
        "five_whys": five_whys,
        "capa": capa,
        "confidence": confidence,
    }


def _augment_root_causes(issue_text: str, existing: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Add deterministic, categorized causes from the issue text to reach 3-5 items.
    """
    # Keyword mapping for simple categorization
    kw_map = [
        (["operator", "staff", "personnel", "human", "training"], "Man", "Operator training gap"),
        (["machine", "equipment", "overheat", "breakdown", "jam"], "Machine", "Machine malfunction"),
        (["procedure", "label", "sop", "method", "process"], "Method", "Procedure not followed"),
        (["material", "batch", "contamination", "impurity"], "Material", "Material contamination"),
        (["calibration", "measurement", "gauge", "sensor"], "Measurement", "Calibration out of spec"),
        (["ambient", "humidity", "temperature", "environment"], "Environment", "Uncontrolled environment"),
    ]
    txt = (issue_text or "").lower()
    causes = list(existing)

    for kws, cat, cause in kw_map:
        if any(k in txt for k in kws) and not any(c.get("category") == cat for c in causes):
            causes.append({"cause": cause, "category": cat})
            if len(causes) >= 3:
                break

    # Ensure at least 3
    while len(causes) < 3:
        causes.append({"cause": "Root cause under investigation", "category": "Method"})

    # Cap at 5
    return causes[:5]


def _ensure_two_capa_entries(capa: List[Dict[str, Any]]) -> None:
    """Mutate capa to ensure exactly two entries: 1 Corrective, 1 Preventive."""
    def _is_type(entry: Dict[str, Any], t: str) -> bool:
        return isinstance(entry.get("type"), str) and entry["type"].lower() == t.lower()

    # Normalize fields & filter invalid
    normalized = []
    for entry in capa:
        t = entry.get("type", "")
        act = entry.get("action", "")
        own = entry.get("owner", "")
        due = entry.get("due_in_days", 14)
        if isinstance(t, str) and isinstance(act, str) and isinstance(own, str):
            try:
                due = int(due)
            except Exception:
                due = 14
            normalized.append({"type": t, "action": act, "owner": own, "due_in_days": due})

    capa.clear()
    # Try to pick one corrective and one preventive from normalized list
    corr = next((e for e in normalized if _is_type(e, "Corrective")), None)
    prev = next((e for e in normalized if _is_type(e, "Preventive")), None)

    if not corr:
        corr = {
            "type": "Corrective",
            "action": "Containment and immediate correction at point of failure",
            "owner": "QA",
            "due_in_days": 7,
        }
    if not prev:
        prev = {
            "type": "Preventive",
            "action": "Review and update SOP; retrain affected operators",
            "owner": "Operations",
            "due_in_days": 14,
        }

    capa.extend([corr, prev])


# -----------------------------
# Public entrypoint
# -----------------------------
def generate_rca_with_llm(issue_text: str,
                          mode: str = "local",
                          max_retries: int = 2,
                          temperature: float = 0.0) -> Dict[str, Any]:
    """
    Generate RCA using preferred backend.
    Modes:
      - "openai": use OpenAI chat completion (if OPENAI_API_KEY available)
      - "api_hf": use Hugging Face Inference API (if HUGGINGFACE_API_KEY available)
      - "local_llm": try local Hugging Face pipeline via transformers + (optional) LangChain/CrewAI
      - "local": deterministic rule-based RCA (no external deps, always works)
    Returns: dict matching the expected schema keys (validated & coerced).
    Raises: LLMRCAException only if a chosen mode is misconfigured (won’t be raised for "local").
    """
    issue_text = str(issue_text or "").strip()
    if not issue_text:
        # Always return a valid structure even for empty text in "local" mode
        if mode == "local":
            return _deterministic_local_rca(issue_text)
        raise LLMRCAException("issue_text is empty.")

    if mode == "openai":
        try:
            return _openai_rca(issue_text, max_retries=max_retries, temperature=temperature)
        except LLMRCAException:
            # Fall through to deterministic local
            return _deterministic_local_rca(issue_text)

    if mode == "api_hf":
        try:
            return _huggingface_api_rca(issue_text)
        except LLMRCAException:
            return _deterministic_local_rca(issue_text)

    if mode == "local_llm":
        try:
            return _local_llm_rca(issue_text, temperature=temperature)
        except LLMRCAException:
            return _deterministic_local_rca(issue_text)

    # Default: deterministic local (no external calls, guaranteed valid)
    return _deterministic_local_rca(issue_text)


# -----------------------------
# Backends
# -----------------------------
def _openai_rca(issue_text: str,
                model: str = "gpt-4o-mini",
                max_retries: int = 2,
                temperature: float = 0.0) -> Dict[str, Any]:
    """OpenAI backend. Requires OPENAI_API_KEY and openai package."""
    api_key = _get_api_key("openai")
    if not api_key:
        raise LLMRCAException("OPENAI_API_KEY not found.")

    try:
        import openai  # type: ignore
    except Exception:
        raise LLMRCAException("openai package not installed.")

    openai.api_key = api_key
    prompt = PROMPT_TEMPLATE.format(issue_text=issue_text)
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a manufacturing quality expert."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
                temperature=temperature,
            )
            content = resp["choices"][0]["message"]["content"]
            json_str = _sanitize_llm_text_to_json(content)
            parsed = json.loads(json_str)
            return _validate_and_coerce_schema(parsed, issue_text)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))

    raise LLMRCAException(f"OpenAI RCA failed after retries. Last error: {last_err}")


def _huggingface_api_rca(issue_text: str,
                         model: str = "MODEL_NAME",
                         timeout: int = 30) -> Dict[str, Any]:
    """
    Hugging Face Inference API backend. Requires requests + HUGGINGFACE_API_KEY.
    NOTE: Replace MODEL_NAME with an instruction-tuned model that reliably outputs JSON.
    """
    api_key = _get_api_key("huggingface")
    if not api_key:
        raise LLMRCAException("HUGGINGFACE_API_KEY not found.")
    if requests is None:
        raise LLMRCAException("requests package not available.")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": PROMPT_TEMPLATE.format(issue_text=issue_text),
        "options": {"wait_for_model": True},
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
        res.raise_for_status()
        data = res.json()

        # Some HF models return a list of {"generated_text": "..."} objects
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            content = data[0]["generated_text"]
        else:
            # Best-effort: stringify and parse JSON block from response
            content = json.dumps(data)

        json_str = _sanitize_llm_text_to_json(content)
        parsed = json.loads(json_str)
        return _validate_and_coerce_schema(parsed, issue_text)

    except Exception as e:
        raise LLMRCAException(f"Hugging Face API RCA failed: {e}")


def _local_llm_rca(issue_text: str,
                   model_name: str = "distilgpt2",
                   temperature: float = 0.0) -> Dict[str, Any]:
    """
    Attempt to use a local HF model via transformers + (optional) LangChain/CrewAI.
    WARNING: Small generic models rarely emit strict JSON. We still sanitize/validate.
    If unavailable or malformed, we fall back to deterministic RCA.
    """
    if hf_pipeline is None or PromptTemplate is None or HuggingFacePipeline is None or Agent is None:
        raise LLMRCAException("Local LLM pipeline dependencies not available.")

    try:
        gen = hf_pipeline("text-generation", model=model_name)
        llm = HuggingFacePipeline(pipeline=gen)
        template = PromptTemplate(input_variables=["issue_text"], template=PROMPT_TEMPLATE)

        agent = Agent(
            role="RCA Specialist",
            goal="Identify root causes and CAPA with valid JSON only",
            backstory="You are a manufacturing QA expert.",
            allow_delegation=False,
            llm=llm,
        )
        task = Task(description=template.format(issue_text=issue_text), agent=agent)
        crew = Crew(tasks=[task], agents=[agent], process=Process.sequential)
        result = crew.kickoff()

        json_str = _sanitize_llm_text_to_json(str(result))
        parsed = json.loads(json_str)
        return _validate_and_coerce_schema(parsed, issue_text)
    except Exception as e:
        raise LLMRCAException(f"Local LLM pipeline failed: {e}")


def _deterministic_local_rca(issue_text: str) -> Dict[str, Any]:
    """
    Fully deterministic, dependency-free RCA. Always returns valid structure.
    This is the safest local mode and recommended as default fallback.
    """
    root_causes = _augment_root_causes(issue_text, existing=[])
    five_whys = [
        f"Why1: Describe how the issue occurred in '{issue_text[:60]}'",
        "Why2: Identify immediate cause at the process step",
        "Why3: Identify systemic/latent cause enabling the immediate cause",
        "Why4: Check gaps in SOP, training, or equipment capability",
        "Why5: Confirm organizational or environmental constraints",
    ]
    capa: List[Dict[str, Any]] = []
    _ensure_two_capa_entries(capa)

    return {
        "root_causes": root_causes,
        "five_whys": five_whys,
        "capa": capa,
        "confidence": "medium",
    }
