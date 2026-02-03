import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset
from datasets.pk_sampler import PKSampler
from loss_functions.triplet_loss import TripletLossWithMining


# ---------------------------
# Config
# ---------------------------
TRAIN_DIR = "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/train/"
VAL_DIR = "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/val/"
CHECKPOINT_DIR = "experiments/checkpoints"

EMBEDDING_DIM = 256
P = 16                  # identities per batch
K = 4                   # images per identity
BATCH_SIZE = P * K
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
MARGIN = 0.2
NUM_WORKERS = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------
# Datasets
# ---------------------------
train_dataset = MuzzleDataset(
    root_dir=TRAIN_DIR,
    transform=train_transform
)

val_dataset = MuzzleDataset(
    root_dir=VAL_DIR,
    transform=val_transform
)

print(f"Train images: {len(train_dataset)}")
print(f"Train identities: {len(set(train_dataset.labels))}")

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
    num_workers=NUM_WORKERS,
    pin_memory=False  # MPS does not use pinned memory
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
# Optimizer
# ---------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ---------------------------
# Loss
# ---------------------------
criterion = TripletLossWithMining(margin=MARGIN)


# ---------------------------
# Training Loop
# ---------------------------
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / max(1, num_batches)


    # ---------------------------
    # Validation (loss only)
    # ---------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / max(1, len(val_loader))

    print(
        f"[Epoch {epoch}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    # ---------------------------
    # Checkpoint
    # ---------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "embedding_dim": EMBEDDING_DIM
            },
            ckpt_path
        )
        print(f"✓ Saved best model at epoch {epoch}")
