import os
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import torch

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
PREPROCESSED = os.path.join(BASE_DIR, "preprocessed")
DATASET = os.path.join(BASE_DIR, "dataset.json")


AUDIO_OUT = os.path.join(BASE_DIR, "audio_embeddings.npy")
TEXT_OUT = os.path.join(BASE_DIR, "text_embeddings.npy")
META_OUT = os.path.join(BASE_DIR, "metadata.json")

# --------------------------
# TEXT embedding → 1536 dims
# then convert to 768 dims
# --------------------------
def text_to_emb(text):
    # Get embedding from OpenAI (3072-d)
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding

    emb = np.array(emb, dtype=np.float32)

    dim = emb.shape[0]  # likely 3072

    # Reduce to 768 dims (Wav2Vec2 output)
    target_dim = 768

    if dim % target_dim != 0:
        raise ValueError(f"Cannot reshape from {dim} → {target_dim}, incompatible")

    factor = dim // target_dim   # = 4 for 3072→768

    emb = emb.reshape(target_dim, factor).mean(axis=1)

    return emb



# -----------------------------------------
# LOAD DATASET.JSON
# -----------------------------------------
with open(DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} items from dataset.json")


# -----------------------------------------
# LOAD Wav2Vec2
# -----------------------------------------
print("Loading Wav2Vec2 model…")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

AUDIO_DIM = 768


# -----------------------------------------
# AUDIO LOADING + RESAMPLING
# -----------------------------------------
def load_audio(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        new_len = int(len(audio) * 16000 / sr)
        audio = resample(audio, new_len)

    return audio.astype(np.float32)


# -----------------------------------------
# GENERATE EMBEDDINGS
# -----------------------------------------
audio_embeddings = []
text_embeddings = []
metadata = []

for item in tqdm(dataset, desc="Embedding"):

    file = item["file"]
    folder = "Drums" if item["class"] == "drums" else "Keys"

    base = os.path.splitext(file)[0]

    candidates = [
        os.path.join(PREPROCESSED, folder, base + ".wav"),
        os.path.join(PREPROCESSED, folder, base + ".aiff"),
        os.path.join(PREPROCESSED, folder, base + ".mp3"),
    ]

    audio_path = None
    for p in candidates:
        if os.path.exists(p):
            audio_path = p
            break

    if audio_path is None:
        print("❌ MISSING:", file)
        continue

    # AUDIO EMB
    audio = load_audio(audio_path)
    inp = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        out = model(**inp)
        emb = out.last_hidden_state.mean(dim=1).numpy().reshape(-1)

    if emb.shape[0] != AUDIO_DIM:
        raise ValueError("Wrong audio emb shape:", emb.shape)

    audio_embeddings.append(emb)

    # TEXT EMB
    t_emb = text_to_emb(item["clean_description"])
    text_embeddings.append(t_emb)

    metadata.append({
        "id": item["id"],
        "file": file,
        "class": item["class"],
        "description": item["clean_description"],
        "audio_path": audio_path
    })


# -----------------------------------------
# SAVE OUTPUT
# -----------------------------------------
np.save(AUDIO_OUT, np.vstack(audio_embeddings))
np.save(TEXT_OUT, np.vstack(text_embeddings))

with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\n🎉 DONE — embeddings created successfully!")
print("Audio embeddings:", np.vstack(audio_embeddings).shape)
print("Text embeddings:", np.vstack(text_embeddings).shape)
print("Metadata saved:", META_OUT)
