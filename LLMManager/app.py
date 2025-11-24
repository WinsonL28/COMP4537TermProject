# app.py
import time
import json
from flask import Flask, request, jsonify
import threading

from llm import generate_response
from db import create_response_record, save_response

app = Flask(__name__)


def background_process(response_id, character_text, previous_context, action, prompt):
    """Runs LLM generation and saves result to DB."""
    try:
        result = generate_response(character_text, previous_context, action, prompt)
        save_response(response_id, prompt, result)
    except Exception as e:
        print("Error in background processing:", e)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    character = data.get("character", {})
    previous_context = data.get("previous_context", [])
    action = data.get("action", "")
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    character_text = json.dumps(character, indent=2) if isinstance(character, dict) else str(character)

    # Create DB row immediately and get response ID
    response_id = create_response_record()

    # Start background processing
    threading.Thread(
        target=background_process,
        args=(response_id, character_text, previous_context, action, prompt),
        daemon=True
    ).start()

    # Return response number immediately
    return jsonify({"response_id": response_id, "status": "processing"}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
