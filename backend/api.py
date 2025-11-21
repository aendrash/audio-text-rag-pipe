import os
import json
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "src")

FAISS_AUDIO = os.path.join(SRC_DIR, "faiss_audio.index")
FAISS_TEXT  = os.path.join(SRC_DIR, "faiss_text.index")
META        = os.path.join(SRC_DIR, "metadata.json")


# Load FAISS + metadata
audio_index = faiss.read_index(FAISS_AUDIO)
text_index  = faiss.read_index(FAISS_TEXT)

with open(META, "r") as f:
    metadata = json.load(f)

app = FastAPI()

class Query(BaseModel):
    query: str

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# EMBEDDING FUNCTION
def embed_query(text):
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding

    emb = np.array(emb, dtype="float32")
    emb = emb.reshape(768, 2).mean(axis=1)
    return emb.reshape(1, -1)


@app.post("/search")
def search_audio(q: Query):
    q_emb = embed_query(q.query)

    dist_a, idx_a = audio_index.search(q_emb, 5)
    dist_t, idx_t = text_index.search(q_emb, 5)

    scores = {}

    for d, i in zip(dist_a[0], idx_a[0]):
        scores[i] = scores.get(i, 0) + (1 - d) * 0.5

    for d, i in zip(dist_t[0], idx_t[0]):
        scores[i] = scores.get(i, 0) + (1 - d) * 0.5

    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final = []
    for idx, score in results[:5]:
        item = metadata[idx]
        final.append({
            "file": item["file"],
            "class": item["class"],
            "description": item["description"],
            "score": float(score),
            "path": item["audio_path"]
        })

    return {"results": final}
