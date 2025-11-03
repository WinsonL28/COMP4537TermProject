from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Open-Orca/Mistral-7B-OpenOrca"
tokenizer = AutoTokenizer.from_pretrained(model_id)
from transformers import BitsAndBytesConfig


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    # Get prompt from request
    raw_prompt = request.json.get("prompt", "")
    if not raw_prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # Format prompt for instruct-style generation
    prompt = f"### Instruction:\n{raw_prompt}\n\n### Response:"

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and strip prompt
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()

    # Return JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)