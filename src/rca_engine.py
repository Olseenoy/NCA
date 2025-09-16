# rca_engine.py
import os
import json
from llm_rca import run_llm_rca
from snca_rca_module import rule_based_rca_fallback, visualize_fishbone_plotly


def ai_rca_with_fallback(record, processed_df=None, sop_library=None, qc_logs=None,
                         reference_folder="nca/data/", llm_backend="ollama"):
    """
    Orchestrates RCA. First tries AI-powered RCA (LLM + reference folder).
    If that fails, falls back to rule-based RCA.
    """
    issue_text = record.get("issue", "")

    result = {
        "issue": issue_text,
        "analysis": None,
        "root_causes": [],
        "five_whys": [],
        "capa": [],
        "fishbone": {},
        "fishbone_fig": None,
    }

    try:
        # ✅ Try AI-based RCA first
        ai_result = run_llm_rca(
            issue_text=issue_text,
            reference_folder=reference_folder,
            backend=llm_backend,
        )

        if ai_result:
            result.update(ai_result)
        else:
            raise ValueError("AI RCA returned empty result")

    except Exception as e:
        # ⚠️ Fallback to rule-based RCA
        print(f"[RCA Engine] AI RCA failed: {e}, using fallback")
        fallback = rule_based_rca_fallback(issue_text, processed_df)
        result.update(fallback)

    # Build fishbone visualization if data available
    if result.get("fishbone"):
        try:
            result["fishbone_fig"] = visualize_fishbone_plotly(result["fishbone"])
        except Exception as e:
            print(f"[RCA Engine] Fishbone plotting failed: {e}")

    return result

