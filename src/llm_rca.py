# =========================
# llm_rca.py (Auto Gemini→Groq, Fallback Fishbone)
# =========================
import os
import json
import google.generativeai as genai
from langchain_groq import ChatGroq
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
    """Generate a fallback fishbone from root causes / whys if LLM does not provide one."""
    categories = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
    fishbone = {c: [] for c in categories}

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

    # If root causes empty → use 5 Whys
    if not any(fishbone.values()):
        whys = parsed.get("five_whys", [])
        if isinstance(whys, str):
            whys = [whys]
        for w in whys:
            fishbone["Method"].append(w)

    return fishbone


def _run_gemini(prompt: str) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("Missing GEMINI_API_KEY")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).text


def _run_groq(prompt: str) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("Missing GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["issue", "reference"], template=prompt)
    )
    return chain.run(issue="", reference="")  # actual vars injected by caller


def run_llm_rca(issue_text: str, reference_folder="nca/data/"):
    """Run RCA with Gemini first, Groq as fallback. Always returns fishbone."""
    reference_text = load_reference_texts(reference_folder)
    if not issue_text.strip():
        raise ValueError("No issue text provided for RCA")

    prompt_template = (
        "You are an RCA expert. An issue was reported:\n\n"
        "ISSUE: {issue}\n\n"
        "We have past RCA documents:\n{reference}\n\n"
        "Please provide:\n"
        "1. Detailed analysis of the issue.\n"
        "2. Likely root causes (list).\n"
        "3. A 5-Whys analysis (stepwise).\n"
        "4. CAPA recommendations (structured).\n"
        "5. Fishbone categories (Man, Machine, Method, Material, Measurement, Environment).\n\n"
        "Return the answer in JSON format with keys: analysis, root_causes, five_whys, capa, fishbone."
    )

    prompt = prompt_template.format(issue=issue_text, reference=reference_text)

    response = None
    parsed = {}

    # --- Try Gemini ---
    try:
        response = _run_gemini(prompt)
        parsed = json.loads(response)
    except Exception as e:
        print(f"[LLM RCA] Gemini failed: {e}")

        # --- Try Groq fallback ---
        try:
            response = _run_groq(prompt_template)
            parsed = json.loads(response)
        except Exception as e2:
            print(f"[LLM RCA] Groq also failed: {e2}")
            parsed = {"analysis": response or "LLM call failed."}

    # ✅ Ensure fishbone always present
    if "fishbone" not in parsed or not parsed.get("fishbone"):
        parsed["fishbone"] = _generate_fallback_fishbone(parsed)

    return parsed
