# =========================
# rca_engine.py (patched with Gemini & Groq)
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
import requests
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
                         reference_folder=None, llm_backend="gemini"):
    """
    Run RCA using Gemini, Groq, Ollama, OpenAI, or Hugging Face dynamically.
    Always returns dict with unified structure for Streamlit.
    """
    issue_text = str(record.get("issue", "")).strip()
    if not issue_text:
        return {"error": "No issue text provided."}

    prompt_text = (
        "You are an RCA (Root Cause Analysis) assistant.\n"
        f"Issue: {issue_text}\n\n"
        "Analyze possible root causes and suggest:\n"
        "1. Detailed analysis\n"
        "2. Root cause(s)\n"
        "3. 5-Whys analysis\n"
        "4. CAPA recommendations (type, action, owner, due_in_days)\n"
        "5. Fishbone diagram (Man, Machine, Method, Material, Measurement, Environment)\n\n"
        "Return in JSON with keys: analysis, root_causes, five_whys, capa, fishbone"
    )

    response_text = None
    backend_used = llm_backend

    # ---------------- GEMINI ----------------
    if llm_backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "Gemini API key not set. Please set GEMINI_API_KEY."}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }]
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    response_text = parts[0]["text"]
            if not response_text:
                response_text = json.dumps(data)
        except Exception as e:
            return {"error": f"Gemini failed: {e}"}

    # ---------------- GROQ ----------------
    elif llm_backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"error": "Groq API key not set. Please set GROQ_API_KEY."}
        url = "https://api.groq.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are an RCA expert."},
                {"role": "user", "content": prompt_text}
            ],
            "model": "llama-3.1-8b-instant",  # free Groq model
            "max_tokens": 600,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            response_text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            ) or json.dumps(data)
        except Exception as e:
            return {"error": f"Groq failed: {e}"}

    # ---------------- Existing Backends ----------------
    elif llm_backend == "ollama":
        try:
            llm = Ollama(model="llama2")
            response = llm.invoke(prompt_text)
            response_text = response if isinstance(response, str) else str(response)
        except Exception as e:
            return {"error": f"Ollama failed: {e}"}

    elif llm_backend == "openai":
        try:
            if not os.getenv("OPENAI_API_KEY"):
                return {"error": "OpenAI API key not set."}
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt_text)
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return {"error": f"OpenAI failed: {e}"}

    elif llm_backend == "huggingface":
        try:
            if not os.getenv("HUGGINGFACE_API_KEY"):
                return {"error": "Hugging Face token not set."}
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
            api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
            payload = {"inputs": prompt_text, "parameters": {"max_new_tokens": 500}}
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                response_text = result[0]["generated_text"]
            else:
                response_text = json.dumps(result)
        except Exception as e:
            return {"error": f"Hugging Face failed: {e}"}

    else:
        return {"error": f"Unsupported LLM backend: {llm_backend}"}

    # ---------------- Parse JSON Response ----------------
    parsed = None
    if response_text:
        try:
            parsed = json.loads(response_text)
        except Exception:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

    if parsed is None:
        parsed = {"raw_text": response_text}

    # ---------------- Unified RCA Result ----------------
    return {
        "backend": backend_used,
        "response": response_text,
        "analysis": parsed.get("analysis", parsed.get("raw_text", "")),
        "why_analysis": parsed.get("five_whys", []),
        "root_cause": parsed.get("root_causes", parsed.get("root_cause", "")),
        "capa": parsed.get("capa", []),
        "fishbone": parsed.get("fishbone", {}),
    }


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
