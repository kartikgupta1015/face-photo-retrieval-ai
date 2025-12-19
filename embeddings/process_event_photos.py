import os
from generate_embeddings import get_face_embeddings, save_embeddings

EVENT_PHOTO_DIR = "data/event_photos"

def process_all_event_photos():
    images = [
        f for f in os.listdir(EVENT_PHOTO_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if not images:
        print("No event photos found!")
        return

    for img in images:
        img_path = os.path.join(EVENT_PHOTO_DIR, img)
        print(f"Processing {img_path} ...")

        embeddings = get_face_embeddings(img_path)

        if not embeddings:
            print(f"⚠️ No faces found in {img}")
            continue

        save_embeddings(embeddings, img_path)
        print(f"✅ Saved {len(embeddings)} face embeddings")

    print("\n🎉 All event photos processed!")


if __name__ == "__main__":
    process_all_event_photos()
