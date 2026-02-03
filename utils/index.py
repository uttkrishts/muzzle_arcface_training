import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.embeddings import MuzzleEmbeddingNet


# ---------------------------
# Config
# ---------------------------
CHECKPOINT_PATH = "experiments/checkpoints/best_model.pt"
INDEX_PATH = "experiments/index/muzzle_index.npz"
META_PATH = "experiments/index/metadata.json"

IMAGE_SIZE = 224
EMBEDDING_DIM = 256

os.makedirs("experiments/index", exist_ok=True)


# ---------------------------
# Device (MPS)
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


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
    return transform(img).unsqueeze(0)


def l2_normalize(x):
    return x / np.linalg.norm(x)


# ---------------------------
# Enrollment
# ---------------------------
def enroll(animal_id, image_paths):
    """
    animal_id: str or int
    image_paths: list of muzzle image paths
    """
    embeddings = []

    with torch.no_grad():
        for p in tqdm(image_paths, desc=f"Enrolling {animal_id}"):
            img = load_image(p).to(device)
            emb = model(img)
            embeddings.append(emb.cpu().numpy()[0])

    embeddings = np.stack(embeddings, axis=0)

    # Template = mean embedding
    template = embeddings.mean(axis=0)
    template = l2_normalize(template)

    return template


# ---------------------------
# Build index
# ---------------------------
def build_index(enrollment_dict):
    """
    enrollment_dict = {
        "animal_001": [img1, img2, ...],
        "animal_002": [...]
    }
    """
    vectors = []
    metadata = []

    for animal_id, image_paths in enrollment_dict.items():
        template = enroll(animal_id, image_paths)
        vectors.append(template)
        metadata.append({
            "animal_id": animal_id,
            "num_images": len(image_paths)
        })

    vectors = np.stack(vectors, axis=0)

    np.savez(INDEX_PATH, vectors=vectors)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved index with {len(vectors)} animals")


# ---------------------------
# Load index
# ---------------------------
def load_index():
    data = np.load(INDEX_PATH)
    vectors = data["vectors"]

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return vectors, metadata
