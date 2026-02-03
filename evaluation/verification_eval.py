import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Project root
sys.path.append(os.getcwd())

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset

# ================= CONFIG =================
CHECKPOINT_PATH = "experiments/checkpoints_arcface/best_arcface_model.pt"
DATA_DIR = (
    "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/train"
)
BATCH_SIZE = 64
NUM_WORKERS = 0  # MPS safe
EMBEDDING_DIM = 512
MIN_IMAGES_PER_ID = 2  # required to build template for verification
# =========================================


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def compute_verification_metrics(pos_scores, neg_scores):
    """
    Computes verification metrics (Accuracy, Optimal Threshold).
    """
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # Labels: 1 for positive, 0 for negative
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])

    # Sort indices by score descending
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]

    # Calculate cumulative True Positives (TP) and False Positives (FP)
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)

    total_pos = len(pos_scores)
    total_neg = len(neg_scores)

    # True Negatives (TN) = Total Negatives - False Positives at that point
    tns = total_neg - fps

    # Accuracy = (TP + TN) / Total
    accuracies = (tps + tns) / (total_pos + total_neg)

    best_idx = np.argmax(accuracies)
    best_acc = accuracies[best_idx]
    best_thresh = sorted_scores[best_idx]

    return best_acc, best_thresh, total_pos, total_neg


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Transforms ----------
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ---------- Dataset ----------
    dataset = MuzzleDataset(DATA_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Images: {len(dataset)}")
    print(f"Identities: {len(set(dataset.labels))}")

    # ---------- Model ----------
    model = MuzzleEmbeddingNet(
        embedding_dim=EMBEDDING_DIM,
        pretrained=False,
    ).to(device)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # ---------- Extract embeddings ----------
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            emb = model(imgs)
            emb = l2_normalize(emb)

            all_embeddings.append(emb.cpu())
            all_labels.extend(labels.tolist())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)

    # ---------- Group embeddings by identity ----------
    by_id = defaultdict(list)
    for emb, label in zip(all_embeddings, all_labels):
        by_id[label.item()].append(emb)

    # Filter weak identities
    by_id = {k: v for k, v in by_id.items() if len(v) >= MIN_IMAGES_PER_ID}

    print(f"Usable identities for verification: {len(by_id)}")

    # Precompute static templates for all IDs (used for negative comparisons)
    static_templates = {}
    valid_labels = list(by_id.keys())

    for label, embs in by_id.items():
        stack = torch.stack(embs).to(device)
        mean_emb = l2_normalize(stack.mean(dim=0))
        static_templates[label] = mean_emb

    # Create a matrix of static templates for fast negative checking
    # Map label -> index in matrix
    label_to_idx = {l: i for i, l in enumerate(valid_labels)}
    static_template_matrix = torch.stack([static_templates[l] for l in valid_labels])
    static_template_matrix = static_template_matrix.to(device)

    # ---------- Verification Loop ----------
    pos_scores = []
    neg_scores = []

    for label, embs in tqdm(by_id.items(), desc="Verifying"):
        embs = torch.stack(embs).to(device)  # [N, Dim]

        # For each image of this valid identity
        for i in range(len(embs)):
            query = embs[i]  # [Dim]

            # 1. Positive Pair: Query vs Own Template (Leave-One-Out)
            # We must exclude the query image from its own template
            remaining = [e for j, e in enumerate(embs) if j != i]
            # Since we filtered MIN_IMAGES_PER_ID >= 2, remaining is never empty
            own_template_loo = l2_normalize(torch.stack(remaining).mean(dim=0))

            sim_pos = torch.matmul(query, own_template_loo).item()
            pos_scores.append(sim_pos)

            # 2. Negative Pairs: Query vs All Other Static Templates
            # Compare query against the big matrix
            all_sims = torch.matmul(query, static_template_matrix.T)  # [Num_Identities]

            # We want to exclude the simulation against the own label's static template
            # (Though technically comparing against own static template is also "positive-ish" but biased,
            #  cleanest negative definition is DIFFERENT identity).

            curr_label_idx = label_to_idx[label]

            # Mask out the current label index
            mask = torch.ones(len(valid_labels), dtype=torch.bool, device=device)
            mask[curr_label_idx] = False

            neg_sims = all_sims[mask]
            neg_scores.extend(neg_sims.cpu().tolist())

    # ---------- Results ----------
    print("\nCalculating metrics...")
    best_acc, best_thresh, n_pos, n_neg = compute_verification_metrics(
        pos_scores, neg_scores
    )

    print("\n================ VERIFICATION RESULTS ================")
    print(f"Total Positive Pairs: {n_pos}")
    print(f"Total Negative Pairs: {n_neg}")
    print(f"Best Accuracy:        {best_acc * 100:.2f}%")
    print(f"Optimal Threshold:    {best_thresh:.4f}")
    print("======================================================\n")

    # Interpretation
    print(f"Interpretation: Use a threshold of {best_thresh:.4f}.")
    print(
        "If cosine_similarity(query_emb, template_emb) > threshold, they are likely the SAME identity."
    )


if __name__ == "__main__":
    main()
