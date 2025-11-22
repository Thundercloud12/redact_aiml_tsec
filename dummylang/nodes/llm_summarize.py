# nodes/llm_summarize.py (FAKE GEMINI FOR TESTING)

def run(inputs: dict):
    pred = inputs.get("pred_class")
    conf_h = inputs.get("prob_healthy")
    conf_d = inputs.get("prob_diseased")
    severity = inputs.get("severity")
    recs = inputs.get("recommendations", [])
    shap = inputs.get("shap_path")

    summary = (
        f"TEST SUMMARY — Prediction: {pred}. "
        f"Healthy={conf_h:.2f}, Diseased={conf_d:.2f}. "
        f"Severity={severity}. "
        f"Heatmap saved at: {shap}. "
        f"Recs: {', '.join(recs[:2])}..."
    )

    return {
        "summary": summary
    }
