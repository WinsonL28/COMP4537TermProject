from transformers import LlamaTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import torch

model_id = "Open-Orca/Mistral-7B-OpenOrca"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)