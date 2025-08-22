# src/llm_rca.py
import os
import json
from typing import Dict, Any

OPENAI_KEY = os.getenv('OPENAI_API_KEY')


class LLMRCAException(Exception):
    pass


def _get_openai_client():
    try:
        import openai
    except ImportError:
        raise LLMRCAException('openai package not installed. pip install openai')
    if not OPENAI_KEY:
        raise LLMRCAException('OPENAI_API_KEY not set in environment')
    openai.api_key = OPENAI_KEY
    return openai


PROMPT_TEMPLATE = '''You are an expert Quality Assurance engineer. Given the following non-conformance report, do three things:

1) Provide the most likely root causes (3-5), each short (1 sentence). Map each cause to a Fishbone category (Man/Machine/Method/Material/Measurement/Environment).
2) Provide a 5-Whys chain (list of 5 answers) that logically goes from the reported problem to a deep root cause.
3) Suggest 2 concrete CAPA items (Corrective and Preventive), each with owner role and estimated due timeframe.

Return output as valid JSON with keys: "root_causes" (list of {"cause":..., "category":...}), "five_whys" (list of 5 strings), "capa" (list of {"type":"Corrective"/"Preventive","action":...,"owner":...,"due_in_days":int}).

Issue:
"""{issue_text}"""

Respond only with JSON, no explanatory text.
'''


def generate_rca_with_llm(issue_text: str) -> Dict[str, Any]:
    openai = _get_openai_client()
    prompt = PROMPT_TEMPLATE.format(issue_text=issue_text)
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    content = resp['choices'][0]['message']['content']
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        import re
        m = re.search(r'{.*}', content, re.S)
        if not m:
            raise LLMRCAException('LLM did not return valid JSON')
        data = json.loads(m.group(0))
    return data
