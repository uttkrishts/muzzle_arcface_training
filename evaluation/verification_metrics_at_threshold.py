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

# The threshold to evaluate at.
# You can change this value to check different operating points.
CONFIDENCE_THRESHOLD = 0.85
# =========================================


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def calculate_confusion_matrix(pos_scores, neg_scores, threshold):
    """
    Calculates TP, TN, FP, FN at a specific threshold.
    """
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # Positive samples: score > threshold -> TP, else FN
    tp = np.sum(pos_scores > threshold)
    fn = np.sum(pos_scores <= threshold)

    # Negative samples: score <= threshold -> TN, else FP
    tn = np.sum(neg_scores <= threshold)
    fp = np.sum(neg_scores > threshold)

    return tp, tn, fp, fn


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating at Threshold: {CONFIDENCE_THRESHOLD}")

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

    for label, embs in tqdm(by_id.items(), desc="Computing Scores"):
        embs = torch.stack(embs).to(device)  # [N, Dim]

        # For each image of this valid identity
        for i in range(len(embs)):
            query = embs[i]  # [Dim]

            # 1. Positive Pair: Query vs Own Template (Leave-One-Out)
            remaining = [e for j, e in enumerate(embs) if j != i]
            own_template_loo = l2_normalize(torch.stack(remaining).mean(dim=0))

            sim_pos = torch.matmul(query, own_template_loo).item()
            pos_scores.append(sim_pos)

            # 2. Negative Pairs: Query vs All Other Static Templates
            all_sims = torch.matmul(query, static_template_matrix.T)  # [Num_Identities]

            curr_label_idx = label_to_idx[label]

            # Mask out the current label index
            mask = torch.ones(len(valid_labels), dtype=torch.bool, device=device)
            mask[curr_label_idx] = False

            neg_sims = all_sims[mask]
            neg_scores.extend(neg_sims.cpu().tolist())

    # ---------- Calculate Metrics ----------
    print("\nCalculating metrics...")
    tp, tn, fp, fn = calculate_confusion_matrix(
        pos_scores, neg_scores, CONFIDENCE_THRESHOLD
    )

    print(
        f"\n================ METRICS AT THRESHOLD {CONFIDENCE_THRESHOLD} ================"
    )
    print(f"True Positives (TP):  {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print("------------------------------------------------------")

    total_pos = tp + fn
    total_neg = tn + fp
    total_samples = total_pos + total_neg

    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    tpr = recall  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Total Samples: {total_samples} (Pos: {total_pos}, Neg: {total_neg})")
    print(f"Accuracy:      {accuracy * 100:.2f}%")
    print(f"Precision:     {precision * 100:.2f}%")
    print(f"Recall (TPR):  {recall * 100:.2f}%")
    print(f"F1 Score:      {f1:.4f}")
    print(f"FPR:           {fpr * 100:.4f}%")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
