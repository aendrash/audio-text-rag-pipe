import faiss
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_EMB = os.path.join(BASE_DIR, "audio_embeddings.npy")
TEXT_EMB = os.path.join(BASE_DIR, "text_embeddings.npy")
META = os.path.join(BASE_DIR, "metadata.json")

FAISS_AUDIO = os.path.join(BASE_DIR, "faiss_audio.index")
FAISS_TEXT = os.path.join(BASE_DIR, "faiss_text.index")


# Load embeddings
audio_emb = np.load(AUDIO_EMB).astype("float32")   # (40, 768)
text_emb = np.load(TEXT_EMB).astype("float32")     # (40, 768)

# Correct dimensions
d_audio = audio_emb.shape[1]      # 768
d_text = text_emb.shape[1]        # 768  ← FIXED

# Build FAISS indices
audio_index = faiss.IndexFlatL2(d_audio)
text_index = faiss.IndexFlatL2(d_text)

audio_index.add(audio_emb)
text_index.add(text_emb)

faiss.write_index(audio_index, FAISS_AUDIO)
faiss.write_index(text_index, FAISS_TEXT)

print("Audio index:", d_audio, "dims")
print("Text index:", d_text, "dims")  # will now be 768
print("Saved FAISS indexes!")
