# train_arcface.py
import os
import sys
import subprocess
import shutil
import shlex
import argparse
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

# rclone config (expects a configured remote, default name: gdrive)
RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
# Set extra rclone flags via env if needed (e.g., "--drive-shared-with-me")
# RCLONE_FLAGS = os.environ.get("RCLONE_FLAGS", "--drive-shared-with-me")

# Train/val paths (override via env if you want custom layout)
TRAIN_DIR = os.environ.get("TRAIN_DIR", os.path.join(DATA_ROOT, "train"))
VAL_DIR = os.environ.get("VAL_DIR", os.path.join(DATA_ROOT, "val"))


def _extract_drive_folder_id(url: str) -> str:
    if "/folders/" in url:
        return url.split("/folders/")[1].split("?")[0]
    return url


def ensure_drive_dataset():
    """Download the Google Drive folder into DATA_ROOT if train/val aren't present."""
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR):
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is required to download the Google Drive dataset. "
            "Install it with: sudo apt-get install -y rclone"
        )

    folder_id = os.environ.get("RCLONE_FOLDER_ID") or _extract_drive_folder_id(
        GDRIVE_FOLDER_URL
    )
    # flags = shlex.split(RCLONE_FLAGS) if RCLONE_FLAGS else []

    print("Downloading dataset from Google Drive using rclone...")
    cmd = [
        "rclone",
        "copy",
        f"{RCLONE_REMOTE}:{folder_id}",
        DATA_ROOT,
        "--progress",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "rclone failed to download the folder. "
            "Make sure your rclone remote is configured and has access, "
            "or set RCLONE_FLAGS (e.g., --drive-shared-with-me) and RCLONE_REMOTE."
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


# ---------------------------
# Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Train ArcFace")
parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
parser.add_argument("--model_lr", type=float, default=1e-2, help="Model learning rate")
parser.add_argument("--loss_lr", type=float, default=1e-2, help="Loss learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adamw"],
    help="Optimizer type",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="experiments/checkpoints_arcface",
    help="Checkpoint directory",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

args = parser.parse_args()

CHECKPOINT_DIR = args.output_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EMBEDDING_DIM = 512

BATCH_SIZE = args.batch_size
# PK sampling approximation (P=Batch/K)
K = 4
P = max(1, BATCH_SIZE // K)

EPOCHS = args.epochs
MODEL_LR = args.model_lr
LOSS_LR = args.loss_lr
WEIGHT_DECAY = args.weight_decay

# ArcFace (IMPORTANT: radians, not degrees)
MARGIN_RAD = 0.50
SCALE = 64

NUM_WORKERS = 0  # MPS safe

# ---------------------------
# Device
# ---------------------------
# if torch.cuda.is_available():
device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

print(f"Using device: {device}")
# if device.type == "cuda":
#     print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ---------------------------
# Transforms
# ---------------------------
train_transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------------------
# Dataset
# ---------------------------
ensure_drive_dataset()
train_dataset = MuzzleDataset(TRAIN_DIR, transform=train_transform)
val_dataset = MuzzleDataset(VAL_DIR, transform=val_transform)

num_classes = len(set(train_dataset.labels))
print(f"Train images: {len(train_dataset)}")
print(f"Train identities: {num_classes}")

# ---------------------------
# Sampler & Loader
# ---------------------------
train_sampler = PKSampler(labels=train_dataset.labels, P=P, K=K)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    # sampler=train_sampler,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# ---------------------------
# Model
# ---------------------------
model = MuzzleEmbeddingNet(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)

# ---------------------------
# ArcFace Loss
# ---------------------------
arcface = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=EMBEDDING_DIM,
    margin=MARGIN_RAD,
    scale=SCALE,
).to(device)

loss_optimizer = torch.optim.SGD(arcface.parameters(), lr=LOSS_LR, momentum=0.9)

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=MODEL_LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MODEL_LR, weight_decay=WEIGHT_DECAY
    )

# Learning Rate Schedulers
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60], gamma=0.1
)
loss_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    loss_optimizer, milestones=[20, 40, 60], gamma=0.1
)


# ---------------------------
# Verification Metrics
# ---------------------------
def compute_eer_auc(embeddings, labels):
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print(
            "Warning: NaN/Inf embeddings detected in validation. Returning trivial metrics."
        )
        return 1.0, 0.5

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

    if len(np.unique(y_true)) < 2:
        print(
            "Warning: Only one class present in validation pair set. Returning trivial metrics."
        )
        return 1.0, 0.5

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

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at Epoch {epoch}. Skipping step.")
            loss_optimizer.zero_grad()
            optimizer.zero_grad()
            continue

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=5.0)

        optimizer.step()
        loss_optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    loss_scheduler.step()

    eer, roc_auc = validate(model, val_loader)

    print(
        f"[Epoch {epoch}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"LR: {current_lr:.9f} | "
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
                "embedding_dim": EMBEDDING_DIM,
            },
            os.path.join(CHECKPOINT_DIR, "best_arcface_model.pt"),
        )
        print(f"✓ Saved best model to {CHECKPOINT_DIR} (EER={eer*100:.2f}%)")
