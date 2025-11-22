# nodes/report_output.py
import json
import os
OUT_DIR = "/kaggle/working/agri_reports"
os.makedirs(OUT_DIR, exist_ok=True)

def run(inputs: dict):
    # assemble final report
    report = {
        "image_path": inputs.get("image_path"),
        "prediction": inputs.get("pred_class"),
        "prob_healthy": inputs.get("prob_healthy"),
        "prob_diseased": inputs.get("prob_diseased"),
        "severity": inputs.get("severity"),
        "recommendations": inputs.get("recommendations"),
        "shap_heatmap": inputs.get("shap_path"),
        "summary": inputs.get("summary"),
    }
    out_path = os.path.join(OUT_DIR, f"report_{os.path.basename(report['image_path'])}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return {"report_path": out_path, "report": report}
