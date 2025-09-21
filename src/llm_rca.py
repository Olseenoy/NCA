# =========================
# llm_rca.py (with Fallback Fishbone)
# =========================
import os
import json
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def load_reference_texts(reference_folder: str):
    """Load past RCA documentation from the reference folder."""
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


def _generate_fallback_fishbone(parsed: dict) -> dict:
    """
    Generate a fallback fishbone from root causes / whys if LLM does not provide one.
    """
    categories = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
    fishbone = {c: [] for c in categories}

    # Use root causes
    root_causes = parsed.get("root_causes", [])
    if isinstance(root_causes, str):
        root_causes = [root_causes]

    for cause in root_causes:
        c_lower = cause.lower()
        if any(word in c_lower for word in ["operator", "staff", "training", "human", "error"]):
            fishbone["Man"].append(cause)
        elif any(word in c_lower for word in ["machine", "equipment", "motor", "seal", "pump", "calibration"]):
            fishbone["Machine"].append(cause)
        elif any(word in c_lower for word in ["method", "process", "procedure", "handling"]):
            fishbone["Method"].append(cause)
        elif any(word in c_lower for word in ["material", "film", "raw", "supply", "milk", "packaging"]):
            fishbone["Material"].append(cause)
        elif any(word in c_lower for word in ["measure", "test", "qc", "inspection", "gauge"]):
            fishbone["Measurement"].append(cause)
        elif any(word in c_lower for word in ["environment", "humidity", "temperature", "dust", "floor", "lighting"]):
            fishbone["Environment"].append(cause)
        else:
            fishbone["Method"].append(cause)  # default bucket

    # Also use 5-Whys hints if root causes were empty
    if not any(fishbone.values()):
        whys = parsed.get("five_whys", [])
        if isinstance(whys, str):
            whys = [whys]
        for w in whys:
            fishbone["Method"].append(w)

    return fishbone


def run_llm_rca(issue_text: str, reference_folder="nca/data/", backend="gemini"):
    """
    Runs RCA using Gemini, Groq, or Ollama + LangChain.
    Returns dict with RCA analysis, 5-Whys, CAPA, and fishbone data.
    """
    reference_text = load_reference_texts(reference_folder)

    if not issue_text.strip():
        raise ValueError("No issue text provided for RCA")

    # Prompt stays flexible (we won't enforce JSON too hard here since we have fallback)
    prompt_template = (
        "You are an RCA expert. An issue was reported:\n\n"
        "ISSUE: {issue}\n\n"
        "We have past RCA documents:\n{reference}\n\n"
        "Please provide:\n"
        "1. Detailed analysis of the issue.\n"
        "2. Likely root causes (list).\n"
        "3. A 5-Whys analysis (stepwise).\n"
        "4. CAPA recommendations (structured).\n"
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

    # ✅ Fallback fishbone
    if "fishbone" not in parsed or not parsed.get("fishbone"):
        parsed["fishbone"] = _generate_fallback_fishbone(parsed)

    return parsed
