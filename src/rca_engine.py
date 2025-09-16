# =========================
# rca_engine.py
# =========================

import os
import pandas as pd
import warnings
import faiss
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
                         reference_folder=None, llm_backend="ollama"):
    issue_text = record.get("issue", "No issue text provided.")

    # Load and index reference docs
    reference_folder = reference_folder or "NCA/data"
    reference_docs = load_reference_files(reference_folder)
    if not reference_docs:
        return {"error": f"No reference documents found in folder: {reference_folder}"}

    index, embeddings, docs = build_faiss_index(reference_docs)

    # Embed issue
    issue_vec = embedder.encode([issue_text], convert_to_numpy=True)
    D, I = index.search(issue_vec, k=min(3, len(docs)))  # Top 3 matches
    retrieved_context = "\n\n".join([docs[i] for i in I[0]])

    # Setup LLM (Ollama local backend)
    if llm_backend == "ollama":
        llm = Ollama(model="llama2")  # Local Ollama model
        prompt_template = PromptTemplate(
            input_variables=["issue", "context"],
            template="""
You are an RCA (Root Cause Analysis) assistant.

Issue: {issue}

Relevant past RCA cases:
{context}

1. Perform a 5 WHY analysis for this issue.
2. Identify the most probable Root Cause.
3. Suggest CAPA (Corrective and Preventive Actions).
4. Provide a fishbone diagram structure (JSON with categories: Methods, Machines, People, Materials, Environment, Measurement).

Respond in JSON with keys: why_analysis, root_cause, capa, fishbone.
"""
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(issue=issue_text, context=retrieved_context)
    else:
        response = {"error": f"Unsupported LLM backend: {llm_backend}"}

    return response


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
