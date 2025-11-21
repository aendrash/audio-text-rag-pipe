import os
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_BASE = os.path.join(BASE_DIR, "sounds")
OUTPUT_BASE = os.path.join(BASE_DIR, "preprocessed")

SR = 32000  # CLAP recommended sample rate

folders = ["Drums", "Keys"]

for folder in folders:
    input_path = os.path.join(INPUT_BASE, folder)
    output_path = os.path.join(OUTPUT_BASE, folder)
    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):

        # Support wav, aiff, mp3
        if not file.lower().endswith((".wav", ".aiff", ".mp3")):
            continue

        file_input = os.path.join(input_path, file)

        # Always output WAV
        file_output = os.path.splitext(os.path.join(output_path, file))[0] + ".wav"

        try:
            # Load audio (librosa supports wav + aiff + mp3)
            audio, _ = librosa.load(file_input, sr=SR, mono=True)

            # Normalize
            audio = audio / max(0.001, audio.max())

            # Save as WAV
            sf.write(file_output, audio, SR)

            print(f"Processed → {file_output}")

        except Exception as e:
            print(f"❌ ERROR processing {file_input} → {e}")

print("\n🎉 Audio preprocessing complete!")
