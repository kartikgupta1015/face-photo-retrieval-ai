import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_event_embeddings():
    database = []

    for file in os.listdir(EMB_DIR):
        if file.endswith(".npy"):
            data = np.load(
                os.path.join(EMB_DIR, file),
                allow_pickle=True
            ).item()
            database.append(data)

    return database


# 👉 STEP C GOES HERE 👇
def find_matching_photos(query_embedding, database, threshold=0.75):
    matched_photos = {}

    for item in database:
        score = cosine_similarity(query_embedding, item["embedding"])

        if score >= threshold:
            photo = item["photo"]
            matched_photos[photo] = max(
                matched_photos.get(photo, 0),
                score
            )

    return sorted(
        matched_photos.items(),
        key=lambda x: x[1],
        reverse=True
    )


if __name__ == "__main__":
    db = load_event_embeddings()

    query = np.load(
        "data/embeddings/test_face0.npy",
        allow_pickle=True
    ).item()["embedding"]

    results = find_matching_photos(query, db)

    print("Matched photos:")
    for photo, score in results:
        print(photo, score)
