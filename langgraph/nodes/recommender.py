# nodes/recommender.py
def run(inputs: dict):
    """
    Simple rule-based suggestions by predicted class name.
    Inputs: pred_class (Healthy/Diseased), prob_healthy, maybe pred_raw
    """
    pred_class = inputs.get("pred_class", "Unknown")
    prob = inputs.get("prob_healthy", 0.0)
    recommendations = []
    severity = "Unknown"
    if pred_class.lower() == "healthy":
        recommendations.append("Plant appears healthy. Maintain regular monitoring and good irrigation practices.")
        severity = "None"
    else:
        # try to map disease-specific advice if class_name includes disease text
        disease_name = inputs.get("disease_name", None)  # optional multi-class support
        # generic safe agricultural advice - non-medical
        recommendations += [
            "Isolate affected plants to prevent spread.",
            "Inspect neighboring plants for similar symptoms.",
            "Remove severely affected leaves and dispose safely away from field.",
            "Consider organic remedies where safe (e.g., neem oil) and consult local extension services for pesticides."
        ]
        # severity heuristic
        conf = inputs.get("prob_diseased", 1.0)
        if conf > 0.9:
            severity = "Severe"
        elif conf > 0.7:
            severity = "Moderate"
        else:
            severity = "Mild"
    return {"recommendations": recommendations, "severity": severity}
