import time
import json
from flask import Flask, request, jsonify
import threading

from llm import generate_response_from_json
from db import create_response_record, save_response, get_connection

app = Flask(__name__)

# waitress-serve --listen=0.0.0.0:5000 app:app

def background_process(data, response_id, story_session_id):
    """Runs LLM generation and saves result to DB."""
    try:
        result = generate_response_from_json(data)
        save_response(response_id, result)
    except Exception as e:
        print("Error in background processing:", e)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 401

    new_prompt = data.get("new_prompt")
    story_session_id = int(data.get("story_session_id"))

    # Create DB row immediately and get response ID
    response_id = create_response_record(story_session_id, new_prompt)

    # Start background processing
    threading.Thread(
        target=background_process,
        args=(data, response_id, story_session_id),
        daemon=True
    ).start()

    # Return response number immediately
    return jsonify({"response_id": response_id, "status": "processing"}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
