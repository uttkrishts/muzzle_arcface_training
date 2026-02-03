import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset

# Config
CHECKPOINT_PATH = "experiments/checkpoints_arcface/best_arcface_model.pt"
DATA_DIR = (
    "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/test"
)
BATCH_SIZE = 64
NUM_WORKERS = 0  # MPS safe
EMBEDDING_DIM = 512


def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    test_dataset = MuzzleDataset(DATA_DIR, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(
        f"Test Set: {len(test_dataset)} images, {len(set(test_dataset.labels))} classes"
    )

    # 2. Load Model
    model = MuzzleEmbeddingNet(embedding_dim=EMBEDDING_DIM, pretrained=False).to(device)

    # 3. Load Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Use weights_only=False because of local checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ArcFace centers (W)
    if "arcface_state_dict" not in checkpoint:
        print("Error: arcface_state_dict not found in checkpoint")
        return

    W = checkpoint["arcface_state_dict"]["W"].to(device)  # shape [256, 97]

    print(f"Loaded model and ArcFace centers (Shape: {W.shape})")

    # 4. Evaluation Loop
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Get embeddings (already normalized by model forward)
            embeddings = model(imgs)  # (B, 256)

            # Normalize weights (ArcFace requirement)
            # W is (256, 97)
            # Normalize along the embedding dimension (dim=0)
            W_norm = F.normalize(W, p=2, dim=0)

            # Compute Logits
            # (B, 256) @ (256, 97) -> (B, 97)
            logits = torch.matmul(embeddings, W_norm)

            # Calculate Accuracy
            # Top-1
            _, pred_top1 = logits.max(1)
            top1_correct += (pred_top1 == labels).sum().item()

            # Top-5
            _, pred_top5 = logits.topk(5, 1, largest=True, sorted=True)
            pred_top5 = pred_top5.t()
            correct = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

            total += labels.size(0)

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    print("\n------------------------------------------------")
    print(f"Evaluation Results on {DATA_DIR}")
    print(f"Total Images: {total}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    evaluate()
