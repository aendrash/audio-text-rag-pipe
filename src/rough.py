import os
import json

BASE_DIR = r"C:\Users\LEGION\project mis\project\sound+text+rag+pipe\sounds"

OUTPUT_DATASET = r"C:\Users\LEGION\project mis\project\sound+text+rag+pipe\dataset_autogen.json"

folders = {
    "Drums": "drums",
    "Keys": "keys"
}

valid_ext = [".wav", ".aiff", ".mp3"]

dataset = []
id_counter = 1

for folder, cls in folders.items():
    folder_path = os.path.join(BASE_DIR, folder)

    for file in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(file)[1].lower()

        if ext not in valid_ext:
            continue

        dataset.append({
            "id": id_counter,
            "file": file,
            "format": ext.replace(".", ""),  # wav / aiff / mp3
            "class": cls,
            "clean_description": ""  # empty for now
        })

        id_counter += 1

with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"🎉 Auto-generated dataset saved!")
print(f"Total items: {len(dataset)}")
