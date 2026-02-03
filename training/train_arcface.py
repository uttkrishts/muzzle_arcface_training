# train_arcface.py
import os
import sys
import subprocess
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset
from datasets.pk_sampler import PKSampler

from pytorch_metric_learning import losses
from sklearn.metrics import roc_curve, auc

# ---------------------------
# Config
# ---------------------------
# Google Drive dataset (folder link)
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1xUkP2ExT8LWI5JJzHJCUpGLqGoVJC1Ja?usp=drive_link"

# Local dataset root on the cloud machine
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data"))

# Train/val paths (override via env if you want custom layout)
TRAIN_DIR = os.environ.get("TRAIN_DIR", os.path.join(DATA_ROOT, "train"))
VAL_DIR   = os.environ.get("VAL_DIR", os.path.join(DATA_ROOT, "val"))

def ensure_drive_dataset():
    """Download the Google Drive folder into DATA_ROOT if train/val aren't present."""
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    try:
        import gdown
        from gdown.exceptions import FolderContentsMaximumLimitError
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download the Google Drive dataset. "
            "Install it with: pip install gdown"
        ) from exc

    print("Downloading dataset from Google Drive...")
    try:
        gdown.download_folder(url=GDRIVE_FOLDER_URL, output=DATA_ROOT, quiet=False, use_cookies=False)
    except FolderContentsMaximumLimitError:
        # Fall back to CLI with --remaining-ok to bypass 50-file listing limit.
        cmd = [
            sys.executable, "-m", "gdown",
            "--folder", "--remaining-ok",
            GDRIVE_FOLDER_URL,
            "-O", DATA_ROOT
        ]
        print("Drive folder has >50 files; retrying with gdown CLI and --remaining-ok...")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "gdown CLI failed to download the folder. "
                "Try upgrading gdown: pip install -U gdown, "
                "or switch to rclone for large datasets."
            )

    # Handle case where Drive folder contains a single subfolder with train/val
    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR)):
        for entry in os.listdir(DATA_ROOT):
            sub = os.path.join(DATA_ROOT, entry)
            if os.path.isdir(sub):
                candidate_train = os.path.join(sub, "train")
                candidate_val = os.path.join(sub, "val")
                if os.path.isdir(candidate_train) and os.path.isdir(candidate_val):
                    print(f"Found nested dataset folder: {sub}")
                    return

    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR)):
        raise FileNotFoundError(
            "Dataset download completed, but train/val folders were not found. "
            "Set DATA_ROOT/TRAIN_DIR/VAL_DIR env vars to point to your dataset layout."
        )
CHECKPOINT_DIR = "experiments/checkpoints_arcface"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EMBEDDING_DIM = 512

# PK sampling
P = 16
K = 4
BATCH_SIZE = P * K

EPOCHS = 60
MODEL_LR = 3e-4
LOSS_LR  = 1e-3
WEIGHT_DECAY = 1e-4

# ArcFace (IMPORTANT: radians, not degrees)
MARGIN_RAD = 0.45
SCALE = 64

NUM_WORKERS = 0  # MPS safe

# ---------------------------
# Device
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ---------------------------
# Transforms
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ---------------------------
# Dataset
# ---------------------------
ensure_drive_dataset()
train_dataset = MuzzleDataset(TRAIN_DIR, transform=train_transform)
val_dataset   = MuzzleDataset(VAL_DIR, transform=val_transform)

num_classes = len(set(train_dataset.labels))
print(f"Train images: {len(train_dataset)}")
print(f"Train identities: {num_classes}")

# ---------------------------
# Sampler & Loader
# ---------------------------
train_sampler = PKSampler(
    labels=train_dataset.labels,
    P=P,
    K=K
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    # sampler=train_sampler,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ---------------------------
# Model
# ---------------------------
model = MuzzleEmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    pretrained=True
).to(device)

# ---------------------------
# ArcFace Loss
# ---------------------------
arcface = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=EMBEDDING_DIM,
    margin=MARGIN_RAD,
    scale=SCALE
).to(device)

loss_optimizer = torch.optim.SGD(
    arcface.parameters(),
    lr=LOSS_LR,
    momentum=0.9
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=MODEL_LR,
    weight_decay=WEIGHT_DECAY
)

# ---------------------------
# Verification Metrics
# ---------------------------
def compute_eer_auc(embeddings, labels):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    scores = []
    y_true = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            scores.append(sim)
            y_true.append(int(labels[i] == labels[j]))

    scores = np.array(scores)
    y_true = np.array(y_true)

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    return eer, roc_auc

def validate(model, loader):
    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = model(imgs)
            all_embs.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    embs = np.concatenate(all_embs)
    labels = np.concatenate(all_labels)

    return compute_eer_auc(embs, labels)

# ---------------------------
# Training Loop
# ---------------------------
best_eer = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    arcface.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        embeddings = model(imgs)
        loss = arcface(embeddings, labels)

        loss.backward()
        optimizer.step()
        loss_optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    avg_train_loss = running_loss / len(train_loader)

    eer, roc_auc = validate(model, val_loader)

    print(
        f"[Epoch {epoch}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"EER: {eer*100:.2f}% | "
        f"AUC: {roc_auc:.4f}"
    )

    if eer < best_eer:
        best_eer = eer
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "eer": eer,
                "auc": roc_auc,
                "embedding_dim": EMBEDDING_DIM
            },
            os.path.join(CHECKPOINT_DIR, "best_arcface_model_new.pt")
        )
        print(f"✓ Saved best model (EER={eer*100:.2f}%)")
