import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.embeddings import MuzzleEmbeddingNet
from utils.index import load_index


# ---------------------------
# Config
# ---------------------------
CHECKPOINT_PATH = "experiments/checkpoints/best_model.pt"

IMAGE_SIZE = 224
EMBEDDING_DIM = 256

# Use threshold from validate.py (EER or stricter)
DEFAULT_THRESHOLD = 0.70


# ---------------------------
# Device (MPS)
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# Model
# ---------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = MuzzleEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    pretrained=False
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------
# Utils
# ---------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img.to(device)


def l2_normalize(x):
    return x / np.linalg.norm(x)


def cosine_similarity(a, b):
    return np.dot(a, b)


# ---------------------------
# Query embedding
# ---------------------------
def extract_query_embedding(image_paths):
    """
    image_paths: list[str]
    """
    embeddings = []

    with torch.no_grad():
        for p in image_paths:
            img = load_image(p)
            emb = model(img)
            embeddings.append(emb.cpu().numpy()[0])

    embeddings = np.stack(embeddings, axis=0)

    # Query template = mean embedding
    template = embeddings.mean(axis=0)
    template = l2_normalize(template)

    return template


# ---------------------------
# Match
# ---------------------------
def match(query_image_paths, threshold=DEFAULT_THRESHOLD):
    """
    Returns:
        {
            "match": bool,
            "animal_id": str or None,
            "score": float
        }
    """
    query_vec = extract_query_embedding(query_image_paths)

    vectors, metadata = load_index()

    scores = np.dot(vectors, query_vec)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return {
            "match": True,
            "animal_id": metadata[best_idx]["animal_id"],
            "score": best_score
        }
    else:
        return {
            "match": False,
            "animal_id": None,
            "score": best_score
        }
