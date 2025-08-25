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

class LLMRCAException(Exception):
    pass

def _get_api_key(provider="openai") -> str | None:
    if provider == "openai":
        if st:
            try:
                key = st.secrets.get("OPENAI_API_KEY")
                if key:
                    return key
            except Exception:
                pass
        return os.getenv("OPENAI_API_KEY")
    elif provider == "huggingface":
        if st:
            try:
                key = st.secrets.get("HUGGINGFACE_API_KEY")
                if key:
                    return key
            except Exception:
                pass
        return os.getenv("HUGGINGFACE_API_KEY")

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

Be concise. Give 3-5 root_causes, exactly 5 five_whys entries, 2 CAPA entries (one corrective, one preventive) with realistic due_in_days.
Issue:
\"\"\"{issue_text}\"\"\"
Respond only with valid JSON that matches the schema above.
'''

def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        candidate = m.group(0)
        candidate = re.sub(r',\s*}', '}', candidate)
        candidate = re.sub(r',\s*\]', ']', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise LLMRCAException(f"Could not parse JSON from LLM response:\n{text[:300]}...")

def generate_rca_with_llm(issue_text: str, mode: str = "local",
                          max_retries: int = 2, temperature: float = 0.0) -> Dict[str, Any]:
    if mode == "openai":
        try:
            return _openai_rca(issue_text, "gpt-4o-mini", max_retries, temperature)
        except LLMRCAException:
            pass
    if mode == "api_hf":
        try:
            return _huggingface_api_rca(issue_text)
        except LLMRCAException:
            pass
    # Default to local Hugging Face pipeline with CrewAI orchestration
    return _local_hf_crew_rca(issue_text)

def _openai_rca(issue_text: str, model: str, max_retries: int, temperature: float) -> Dict[str, Any]:
    api_key = _get_api_key("openai")
    if not api_key:
        raise LLMRCAException("OPENAI_API_KEY not found.")
    try:
        import openai
    except ImportError as e:
        raise LLMRCAException("openai package not installed.") from e
    openai.api_key = api_key
    prompt = PROMPT_TEMPLATE.format(issue_text=issue_text)
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a manufacturing quality expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=temperature,
            )
            content = resp["choices"][0]["message"]["content"]
            parsed = _parse_json_from_text(content)
            if not all(k in parsed for k in ["root_causes", "five_whys", "capa"]):
                raise LLMRCAException("Incomplete JSON from OpenAI RCA.")
            return parsed
        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(1.0 * attempt)
            continue
    raise LLMRCAException(f"OpenAI RCA failed after retries. Last error: {last_err}")

def _huggingface_api_rca(issue_text: str) -> Dict[str, Any]:
    api_key = _get_api_key("huggingface")
    if not api_key:
        raise LLMRCAException("HUGGINGFACE_API_KEY not found.")
    url = "https://api-inference.huggingface.co/models/MODEL_NAME"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": issue_text}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            raise LLMRCAException(f"Hugging Face RCA error: {data['error']}")
        return {
            "root_causes": [{"cause": "HF RCA cause", "category": "Method"}],
            "five_whys": ["HF Why1", "HF Why2", "HF Why3", "HF Why4", "HF Why5"],
            "capa": [
                {"type": "Corrective", "action": "HF corrective action", "owner": "QA", "due_in_days": 7},
                {"type": "Preventive", "action": "HF preventive action", "owner": "Maintenance", "due_in_days": 14}
            ],
            "confidence": "medium"
        }
    except Exception as e:
        raise LLMRCAException(f"Hugging Face RCA failed: {e}")

def _local_hf_crew_rca(issue_text: str) -> Dict[str, Any]:
    # Setup local model pipeline
    local_pipeline = pipeline("text-generation", model="distilgpt2")
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    # Define prompt
    template = PromptTemplate(
        input_variables=["issue_text"],
        template=PROMPT_TEMPLATE
    )

    # Define CrewAI agents
    rca_agent = Agent(
        role="RCA Specialist",
        goal="Identify root causes and CAPA",
        backstory="You are a manufacturing expert with RCA experience.",
        allow_delegation=False,
        llm=llm
    )

    task = Task(description=template.format(issue_text=issue_text), agent=rca_agent)
    crew = Crew(tasks=[task], agents=[rca_agent], process=Process.sequential)
    result = crew.kickoff()

    return _parse_json_from_text(result)
