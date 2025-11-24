# llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Open-Orca/Mistral-7B-OpenOrca"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)


AI_JOB = """You are a Dungeons & Dragons-style AI Game Master.
Rules (MUST follow exactly):
1) NEVER perform actions or decide for the player. Do not cast spells, move the character, or resolve attacks on the player's behalf.
2) Describe the scene and possible options briefly and vividly. Then STOP.
3) Always end the response by prompting the player for their next action using a short question (examples below).
4) Use the character data to tailor descriptions, but do not apply or deduct any stat values unless the player explicitly chooses an action and you are asked to resolve it.
5) Never reveal you are an AI or break character.

Desired format (few-shot examples):

# Example 1 (combat opportunity)
You see three goblins in the underbrush. One holds a crude torch; another grips a short spear. None have noticed you yet.
What will you do next?

# Example 2 (interaction)
An old woman sits by the path and watches you intently. She taps a small wooden box on her knee.
Your move, adventurer?

Strictly follow these examples. If you ever produce any description that implies the player already acted (e.g. 'you fired', 'you cast'), truncate that part and instead ask for the player’s choice. End.
"""


def build_chatml(character_text, previous_context, action, prompt):
    messages = [
        {"role": "system", "content": AI_JOB.strip()},
        {
            "role": "user",
            "content": f"Character Data:\n{character_text}\n\nPrevious context: {previous_context}\n\nAction: {action}\n\nPrompt: {prompt}"
        }
    ]
    chatml = ""
    for m in messages:
        chatml += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    chatml += "<|im_start|>assistant\n"
    return chatml


def generate_response(character_text, previous_context, action, prompt, max_tokens=400):
    chatml_prompt = build_chatml(character_text, previous_context, action, prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(chatml_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean up any ChatML markers
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in decoded:
        decoded = decoded.split("<|im_end|>")[0].strip()

    return decoded
