import streamlit as st
import openai
import json

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_rca_with_llm(issue_text):
    prompt = f"""
    You are a manufacturing quality expert.
    Analyze this issue and respond in JSON format with:
    - root_cause: string
    - five_whys: list of five strings
    - capa: list of action dictionaries with 'action', 'owner', 'due_days'

    Issue: {issue_text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert in RCA and CAPA."},
                      {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"OpenAI call failed: {e}"}
