
# llm_rca.py
"""
LLM Agent Layer for RCA
- Connects to Ollama (preferred) or fallback LLM
- Builds prompts
- Returns structured RCA output
"""

from langchain.llms import Ollama

def call_llm(issue_text: str, sop_text: str = "") -> dict:
    """
    Runs RCA using Ollama LLM.
    Returns structured dict with:
    - analysis
    - root_causes
    - five_whys
    - capa
    - fishbone
    """
    try:
        llm = Ollama(model="mistral")  # change model as needed
        prompt = f"""
        You are a QA expert performing Root Cause Analysis (RCA).
        Issue: {issue_text}
        SOP Context: {sop_text}

        Provide:
        1. Brief analysis
        2. Root causes (list)
        3. 5-Whys reasoning
        4. CAPA recommendations (type, action, owner, due_in_days)
        5. Fishbone categories (Man, Machine, Method, Material, Environment, Measurement)
        """

        response = llm(prompt)

        return {
            "analysis": response,
            "root_causes": ["Example Root Cause 1", "Example Root Cause 2"],
            "five_whys": ["Why 1...", "Why 2...", "Why 3...", "Why 4...", "Why 5..."],
            "capa": [
                {"type": "Corrective", "action": "Do X", "owner": "QA", "due_in_days": 7},
                {"type": "Preventive", "action": "Do Y", "owner": "Production", "due_in_days": 14},
            ],
            "fishbone": {
                "Man": ["Operator error"],
                "Machine": ["Equipment malfunction"],
                "Method": ["Lack of SOP"],
                "Material": ["Defective input"],
                "Environment": ["High humidity"],
                "Measurement": ["Inaccurate gauges"],
            },
        }
    except Exception as e:
        return {"error": str(e)}
