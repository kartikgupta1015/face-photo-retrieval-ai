from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
import shutil
import os
import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

from search.similarity_search import load_event_embeddings, find_matching_photos

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
SELFIE_DIR = os.path.join(DATA_DIR, "selfies")

app = FastAPI()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Models
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

os.makedirs(SELFIE_DIR, exist_ok=True)


def get_selfie_embedding(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = mtcnn(img_rgb)
    if face is None:
        raise ValueError("No face detected")

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(face)

    return emb.squeeze().cpu().numpy()


@app.post("/find-photos")
async def find_photos(file: UploadFile = File(...)):
    selfie_path = os.path.join(SELFIE_DIR, file.filename)

    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_embedding = get_selfie_embedding(selfie_path)

    database = load_event_embeddings()

    results = find_matching_photos(
        query_embedding,
        database,
        threshold=0.75
    )

    return {
        "matched_photos": [
            {"photo": photo, "score": round(score, 3)}
            for photo, score in results
        ]
    }

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")