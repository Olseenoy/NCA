# =========================
# rca_engine.py
# =========================
import os
import pandas as pd
import warnings
import faiss
import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import json
import re
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from docx import Document
from pypdf import PdfReader

# --- Load embedding model once ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Utility: Load files recursively from reference folder ---
def load_reference_files(reference_folder):
    docs = []
    for root, _, files in os.walk(reference_folder):  # <-- recursively walk subfolders
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                if ext in [".txt", ".log", ".md", ".json", ".csv"]:
                    if ext == ".csv":
                        df = pd.read_csv(fpath)
                        text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
                    else:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    docs.append(text)
                elif ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(fpath)
                    text = "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1))
                    docs.append(text)
                elif ext == ".docx":
                    doc = Document(fpath)
                    text = "\n".join([p.text for p in doc.paragraphs])
                    docs.append(text)
                elif ext == ".pdf":
                    reader = PdfReader(fpath)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    docs.append(text)
                else:
                    warnings.warn(f"Unsupported file type skipped: {fname}")
            except Exception as e:
                warnings.warn(f"Failed to read {fname}: {e}")
    if not docs:
        warnings.warn(f"No reference documents found in folder: {reference_folder}")
    return docs


# --- Utility: Build FAISS index ---
def build_faiss_index(docs):
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, docs


# --- Main RCA Function ---

def ai_rca_with_fallback(record, processed_df=None, sop_library=None, qc_logs=None,
                         reference_folder=None, llm_backend="ollama",
                         openai_key=None, hf_token=None):
    import os
    import json, re
    import requests
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.llms import Ollama

    # Read tokens from environment if not passed
    openai_key = openai_key or os.getenv("OPENAI_API_KEY")
    hf_token = hf_token or os.getenv("HF_API_TOKEN")

    issue_text = str(record.get("issue", "")).strip()
    if not issue_text:
        return {"error": "No issue text provided."}

    prompt = f"You are an RCA assistant.\nIssue: {issue_text}\nSuggest root causes and CAPA."

    response_text = None

    if llm_backend == "ollama":
        llm = Ollama(model="llama2")
        response_text = llm.invoke(prompt)

    elif llm_backend == "openai":
        if not openai_key:
            return {"error": "OpenAI API key not set."}
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

    elif llm_backend == "huggingface":
        if not hf_token:
            return {"error": "Hugging Face token not set."}
        headers = {"Authorization": f"Bearer {hf_token}"}
        api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}
        r = requests.post(api_url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            response_text = result[0]["generated_text"]
        else:
            response_text = json.dumps(result)

    else:
        return {"error": f"Unsupported backend: {llm_backend}"}

    # Try to parse JSON if possible
    try:
        parsed = json.loads(response_text)
    except Exception:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"raw_text": response_text}

    return {"backend": llm_backend, "response": response_text, "parsed": parsed}





# --- Utility: Plot Fishbone diagram ---
def visualize_fishbone_plotly(fishbone_dict):
    categories = list(fishbone_dict.keys())
    fig = go.Figure()
    for i, cat in enumerate(categories):
        causes = fishbone_dict[cat]
        for j, cause in enumerate(causes):
            fig.add_trace(go.Scatter(
                x=[i, i+0.5],
                y=[0, -(j+1)],
                mode="lines+markers+text",
                text=[cat, cause],
                textposition="top center",
                line=dict(color="blue", width=2),
                marker=dict(size=8)
            ))
    fig.update_layout(title="Fishbone Diagram", showlegend=False)
    return fig

