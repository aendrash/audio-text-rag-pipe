import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================
# IMPORTANT: RELATIVE PATHS
# ============================

BASE_DIR = os.path.dirname(__file__)          # → src/
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds") # → src/sounds/
TEXT_DIR = os.path.join(SOUNDS_DIR, "text")   # → src/sounds/text/


def load_text(path):
    return open(path, "r", encoding="utf-8").read()


def list_audio(folder):
    exts = (".wav", ".aiff", ".mp3")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def prompt_for_one_file(filename, raw_text, class_name):
    return f"""
You MUST produce a JSON object for exactly ONE file:

Filename: {filename}
Class: {class_name}

Task:
- Search the raw text for any matching or similar description.
- Use filename keywords when matching.
- If NO matching description exists → clean_description = "".
- NEVER invent details.

Return ONLY JSON:
{{
  "file": "{filename}",
  "format": "{filename.split('.')[-1].lower()}",
  "class": "{class_name}",
  "clean_description": "<1–2 sentences OR empty>"
}}

Raw text:
----------------
{raw_text}
----------------
"""


def process_class(class_name, text_file, folder_path, start_id):
    raw_text = load_text(text_file)
    files = list_audio(folder_path)

    results = []

    for i, file in enumerate(files, start=1):
        prompt = prompt_for_one_file(file, raw_text, class_name)

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = resp.choices[0].message.content.strip()
        clean = output.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(clean)
        except Exception:
            print("⚠ JSON parse error for", file)
            print(clean)
            continue

        data["id"] = start_id + i - 1
        results.append(data)

        print(f"✔ Processed {file}")

    return results


# ======== RUN BOTH CLASSES ========
drums = process_class(
    "drums",
    os.path.join(TEXT_DIR, "Drums.txt"),
    os.path.join(SOUNDS_DIR, "drums"),
    start_id=1
)

keys = process_class(
    "keys",
    os.path.join(TEXT_DIR, "Keys.txt"),
    os.path.join(SOUNDS_DIR, "keys"),
    start_id=101
)

dataset = drums + keys

# Save output in src/
OUT = os.path.join(BASE_DIR, "dataset.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print("🎉 FINAL dataset.json created! Total:", len(dataset))
