# rca_engine.py

import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from PyPDF2 import PdfReader

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utility: Load files from reference folder ---
def load_reference_files(reference_folder):
    docs = []
    for fname in os.listdir(reference_folder):
        fpath = os.path.join(reference_folder, fname)

        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())

        elif fname.endswith(".docx"):
            doc = Document(fpath)
            docs.append("\n".join([p.text for p in doc.paragraphs]))

        elif fname.endswith(".pdf"):
            reader = PdfReader(fpath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)

    return docs

# --- Utility: Build FAISS index ---
def build_faiss_index(docs):
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, docs

# --- Main RCA Function ---
def ai_rca_with_fallback(record, processed_df, sop_library, qc_logs, reference_folder, llm_backend="ollama"):
    issue_text = record.get("issue", "No issue text provided.")

    # Load and index reference docs
    reference_docs = load_reference_files(reference_folder)
    if not reference_docs:
        return {"error": "No reference documents found in folder."}

    index, embeddings, docs = build_faiss_index(reference_docs)

    # Embed issue
    issue_vec = embedder.encode([issue_text], convert_to_numpy=True)
    D, I = index.search(issue_vec, k=3)  # Top 3 matches

    retrieved_context = "\n\n".join([docs[i] for i in I[0]])

    # Setup LLM (Ollama local backend)
    if llm_backend == "ollama":
        llm = Ollama(model="llama2")  # Change to any local Ollama model

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
        response = {
            "error": f"Unsupported LLM backend: {llm_backend}"
        }

    return response
