# nodes/llm_summarize.py
import os
import json
# This node formats a prompt and calls an LLM (Gemini / OpenAI). 
# Replace the `call_gemini` function with your chosen API client and keys.
def call_gemini(prompt: str):
    """
    Minimal placeholder: Replace with actual API call to Gemini/OpenAI.
    Example (pseudo):
       from openai import OpenAI
       client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
       resp = client.responses.create(model="gpt-4o-mini", input=prompt, ...)
       return resp.output_text
    """
    # For now return the prompt snippet to simulate a reply
    return "SIMULATED_SUMMARY: " + prompt[:400]

def run(inputs: dict):
    pred_class = inputs.get("pred_class")
    prob_h = inputs.get("prob_healthy")
    prob_d = inputs.get("prob_diseased")
    shap_path = inputs.get("shap_path")
    recs = inputs.get("recommendations", [])
    severity = inputs.get("severity", "Unknown")
    image_path = inputs.get("image_path", "image")
    # Build a prompt for summarization
    prompt = f"""
You are an agricultural assistant. Analyze the following detection:
- Image: {image_path}
- Prediction: {pred_class}
- Confidence: healthy={prob_h:.3f}, diseased={prob_d:.3f}
- Severity: {severity}
- Recommendations: {json.dumps(recs)}
- SHAP heatmap file path: {shap_path}

Write a short farmer-friendly summary (3-5 lines) that:
1) States whether plant is healthy or diseased
2) Gives a clear action (isolate, inspect, apply organic neem, or monitor)
3) Mentions confidence and where the SHAP heatmap highlights (e.g., "spots near leaf edges")
"""
    # call real LLM here
    llm_out = call_gemini(prompt)
    return {"summary": llm_out, "prompt": prompt}
