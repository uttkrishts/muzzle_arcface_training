import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset


# ---------------------------
# Config
# ---------------------------
TEST_DIR = "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/test/"
CHECKPOINT_PATH = "/Users/taglineinfotechllp/Documents/RnD/CattleSelf/experiments/checkpoints_arcface/best_arcface_model.pt"

EMBEDDING_DIM = 512
BATCH_SIZE = 64
NUM_WORKERS = 0


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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# Dataset
# ---------------------------
dataset = MuzzleDataset(
    root_dir=TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Test images: {len(dataset)}")
print(f"Test identities: {len(set(dataset.labels))}")


# ---------------------------
# Model
# ---------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,weights_only=False)

model = MuzzleEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    pretrained=False
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------
# Extract embeddings
# ---------------------------
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

embeddings = torch.cat(all_embeddings, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()


# ---------------------------
# Pairwise similarity
# ---------------------------
def cosine_similarity(a, b):
    return np.dot(a, b)


genuine_scores = []
impostor_scores = []

N = len(embeddings)

for i in range(N):
    for j in range(i + 1, N):
        score = cosine_similarity(embeddings[i], embeddings[j])

        if labels[i] == labels[j]:
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"Genuine pairs: {len(genuine_scores)}")
print(f"Impostor pairs: {len(impostor_scores)}")


# ---------------------------
# ROC / AUC
# ---------------------------
y_true = np.concatenate([
    np.ones_like(genuine_scores),
    np.zeros_like(impostor_scores)
])

y_scores = np.concatenate([
    genuine_scores,
    impostor_scores
])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC: {roc_auc:.4f}")


# ---------------------------
# Equal Error Rate (EER)
# ---------------------------
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"EER: {eer:.4f}")
print(f"EER Threshold: {eer_threshold:.4f}")


# ---------------------------
# Practical thresholds
# ---------------------------
for target_fpr in [0.01, 0.001]:
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) > 0:
        t = thresholds[idx[-1]]
        print(f"Threshold @ FPR {target_fpr*100:.1f}%: {t:.4f}")

