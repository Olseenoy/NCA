# =========================
# llm_rca.py (patched)
# =========================
import os
import json
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def load_reference_texts(reference_folder: str):
    """
    Load past RCA documentation from the reference folder.
    Returns concatenated string of all file contents.
    """
    if not os.path.exists(reference_folder):
        return ""

    texts = []
    for fname in os.listdir(reference_folder):
        fpath = os.path.join(reference_folder, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".md", ".json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"[LLM RCA] Failed reading {fname}: {e}")

    return "\n\n".join(texts)


def run_llm_rca(issue_text: str, reference_folder="nca/data/", backend="gemini"):
    """
    Runs RCA using Gemini, Groq, or Ollama + LangChain.
    Returns dict with RCA analysis, 5-Whys, CAPA, and fishbone data.
    """
    reference_text = load_reference_texts(reference_folder)

    if not issue_text.strip():
        raise ValueError("No issue text provided for RCA")

    # Build the RCA prompt
    prompt_template = (
        "You are an RCA expert. An issue was reported:\n\n"
        "ISSUE: {issue}\n\n"
        "We have past RCA documents:\n{reference}\n\n"
        "Please provide:\n"
        "1. A detailed analysis of the issue.\n"
        "2. Likely root causes (structured JSON list).\n"
        "3. A 5-Whys analysis (stepwise).\n"
        "4. CAPA recommendations (JSON: type, action, owner, due_in_days).\n"
        "5. Fishbone categories (Man, Machine, Method, Material, Measurement, Environment) with possible causes.\n\n"
        "Return the answer in JSON format with keys: analysis, root_causes, five_whys, capa, fishbone."
    )

    response = None

    # --- Gemini Backend ---
    if backend == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError("Missing GEMINI_API_KEY")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response_obj = model.generate_content(
            prompt_template.format(issue=issue_text, reference=reference_text)
        )
        response = response_obj.text

    # --- Groq Backend ---
    elif backend == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("Missing GROQ_API_KEY")
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["issue", "reference"], template=prompt_template)
        )
        response = chain.run(issue=issue_text, reference=reference_text)

    # --- Ollama (optional legacy) ---
    elif backend == "ollama":
        llm = Ollama(model="mistral")
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["issue", "reference"], template=prompt_template)
        )
        response = chain.run(issue=issue_text, reference=reference_text)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Parse JSON response safely
    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {"analysis": response}

    return parsed
