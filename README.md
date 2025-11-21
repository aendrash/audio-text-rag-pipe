🎧 Audio + Text RAG Search Engine

Retrieve audio samples using natural-language descriptions

This project is a complete RAG pipeline for audio.
It lets you search drum/keys sounds using text queries such as:

“Find me a punchy kick drum”
“Show me a soft key pad sound”
“I need a snare with reverb”

The system embeds audio + text descriptions, stores them in FAISS, and exposes a FastAPI backend with a Streamlit UI frontend.

🚀 Live Demo

Frontend (Streamlit):
👉 https://audio-text-rag-pipe.streamlit.app/

Backend (Render – FastAPI):
👉 https://audio-text-rag-pipe.onrender.com

📁 Project Structure
audio-text-rag-pipe/
│
├── backend/
│   ├── api.py                # FastAPI backend
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   └── ui.py                 # Streamlit UI
│
└── src/
    ├── 1.cleaning.py
    ├── 2.preprocess_audio.py
    ├── 3.create_embeddings.py
    ├── 4.build_faiss.py
    ├── dataset.json
    ├── metadata.json
    ├── audio_embeddings.npy
    ├── text_embeddings.npy
    ├── faiss_audio.index
    ├── faiss_text.index
    ├── sounds/
    │     ├── Drums/
    │     └── Keys/
    └── preprocessed/
          ├── Drums/
          └── Keys/

🧠 How It Works
1️⃣ Cleaning / Dataset Building

Reads text descriptions from your .txt files

Matches audio files with their text descriptions using GPT

Produces dataset.json

2️⃣ Audio Pre-Processing

Loads raw WAV/AIFF/MP3

Resamples everything to 32 kHz

Normalizes audio

Saves to preprocessed/*

3️⃣ Embeddings

Audio embeddings (768-dim)
→ Extracted using facebook/wav2vec2-base-960h

Text embeddings (3072-dim)
→ Reduced to 768-dim by mean-pooling

Saves:

audio_embeddings.npy

text_embeddings.npy

metadata.json

4️⃣ FAISS Indexing

Two separate FAISS indexes:

Index	Dimension	Purpose
faiss_audio.index	768	Audio vector search
faiss_text.index	768	Text vector search
5️⃣ Backend (FastAPI)

/search
→ Accepts text query
→ Embeds query
→ Searches FAISS
→ Returns top results
→ Provides audio URL /audio/<path>

6️⃣ Frontend (Streamlit)

Text input for query

Calls backend

Streams audio using URL

Displays results

🧩 Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here


For Streamlit Cloud:

API_URL="https://audio-text-rag-pipe.onrender.com/search"

🖥️ Run Project Locally
Backend
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000


Open:
👉 http://127.0.0.1:8000/docs

Frontend
cd frontend
streamlit run ui.py

☁️ Deployment Guide
Backend → Render

New Web Service

Root Directory: backend

Build Command:

pip install -r requirements.txt


Start Command:

uvicorn api:app --host 0.0.0.0 --port $PORT


Add environment variable:
OPENAI_API_KEY

Frontend → Streamlit Cloud

Add secret:

API_URL="https://audio-text-rag-pipe.onrender.com/search"


Then deploy normally.

🔥 Features

✔ Natural-language audio search
✔ Audio + text RAG pipeline
✔ Wav2Vec2 embeddings
✔ GPT-powered text cleaning
✔ FAISS fast retrieval
✔ Streamlit UI
✔ Render backend hosting
✔ Works fully server-side

🧪 Future Improvements

Add spectrogram visualization

Add similarity heatmaps

Add multi-class audio categories

Add CLAP embeddings

Add file upload for user input audio

🤝 Contributing

Pull requests are welcome!
If you’d like new features, open an Issue.

📜 License

MIT License.
