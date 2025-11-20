import time
import json
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# waitress-serve --listen=0.0.0.0:5000 app:app

model_id = "Open-Orca/Mistral-7B-OpenOrca"
tokenizer = AutoTokenizer.from_pretrained(model_id)
from transformers import BitsAndBytesConfig


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def explore():
    start_time = time.time()

    # --- Get prompt data ---
    character = request.json.get("character", {})
    
    AI_job = """You are a Dungeons & Dragons-style AI Game Master.
    Rules (MUST follow exactly):
    1) NEVER perform actions or decide for the player. Do not cast spells, consume MP, move the character, or resolve attacks on the player's behalf.
    2) Describe the scene and possible options briefly and vividly. Then STOP.
    3) Always end the response by prompting the player for their next action using a short question (examples below).
    4) Use the character data to tailor descriptions, but do not apply or deduct any stat values unless the player explicitly chooses an action and you are asked to resolve it.
    5) Never reveal you are an AI or break character.

    Desired format (few-shot examples):

    # Example 1 (combat opportunity, correct)
    You see three goblins in the underbrush. One holds a crude torch; another grips a short spear. None have noticed you yet.
    Possible options: 1) Fire Piercing Arrow at the lead goblin (cost 10 MP), 2) Hide and observe, 3) Call out and attempt to parley.
    What will you do next?

    # Example 2 (interaction, correct)
    An old woman sits by the path and watches you intently. She taps a small wooden box on her knee.
    Possible options: 1) Ask about the box, 2) Offer help, 3) Ignore and pass by.
    Your move, adventurer?

    Strictly follow these examples. If you ever produce any description that implies the player already acted (e.g. 'you fired', 'you cast'), truncate that part and instead ask for the player’s choice. End.
    """

    
    
    previous_context = request.json.get("previous_context", [])
    action = request.json.get("action", "")
    raw_prompt = request.json.get("prompt", "")

    if not raw_prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # --- Convert character dict into readable text for the AI ---
    if isinstance(character, dict):
        character_text = json.dumps(character, indent=2)
    else:
        character_text = str(character)

    # --- Build ChatML messages ---
    messages = [
        {"role": "system", "content": AI_job.strip()},
        {"role": "user", "content": f"Character Data:\n{character_text}\n\nPrevious context: {previous_context}\n\nAction: {action}\n\n{raw_prompt}"}
    ]

    # --- Convert to ChatML ---
    chatml_prompt = ""
    for m in messages:
        chatml_prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    chatml_prompt += "<|im_start|>assistant\n"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # Check if GPU or CPU

    
    # --- Tokenize and generate ---
    inputs = tokenizer(chatml_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in decoded:
        decoded = decoded.split("<|im_end|>")[0].strip()

    # --- Extract scene and options from AI text ---
    description = decoded
    options = []
    if "Possible options:" in decoded:
        desc, opts_text = decoded.split("Possible options:", 1)
        description = desc.strip()
        for line in opts_text.strip().split("\n"):
            if ")" in line:
                options.append(line.split(")",1)[1].strip())

    # Append to previous context
    previous_context.append({
        "action": action,
        "system_response": description
    })

    elapsed_time = round(time.time() - start_time, 3)

    # Return structured JSON
    return jsonify({
        "generation_time_sec": elapsed_time,
        "character": character,
        "previous_context": previous_context,
        "current_scene": {
            "description": description,
            "possible_options": options
        }
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)