import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = mtcnn(img_rgb)

    if faces is None:
        return []

    embeddings = []

    for face in faces:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(face)
        embeddings.append(emb.squeeze().cpu().numpy())

    return embeddings
    

def save_embeddings(embeddings, image_path):
    os.makedirs("data/embeddings", exist_ok=True)
    base = os.path.basename(image_path)

    for i, emb in enumerate(embeddings):
        np.save(
            f"data/embeddings/{base}_face{i}.npy",
            {
                "embedding": emb,
                "photo": image_path
            },
            allow_pickle=True
        )


if __name__ == "__main__":
    image_path = "data/event_photos/test.jpg"

    embs = get_face_embeddings(image_path)
    print(f"Detected {len(embs)} faces")

    # 👉 CALL IT HERE 👇
    save_embeddings(embs, image_path)

    if embs:
        print("Embedding shape:", embs[0].shape)
