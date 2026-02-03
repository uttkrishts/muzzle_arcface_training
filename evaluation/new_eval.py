import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

# Project root
sys.path.append(os.getcwd())

from models.embeddings import MuzzleEmbeddingNet
from datasets.muzzle_datasets import MuzzleDataset

# ================= CONFIG =================
CHECKPOINT_PATH = "experiments/checkpoints_arcface/best_arcface_model.pt"
DATA_DIR = "/Users/taglineinfotechllp/Documents/RnD/cattle_resnet/data/finetuning_dataset/test"
BATCH_SIZE = 64
NUM_WORKERS = 0  # MPS safe
EMBEDDING_DIM = 512
MIN_IMAGES_PER_ID = 2  # required to build template
# =========================================


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Transforms ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

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

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
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
    by_id = {
        k: v for k, v in by_id.items()
        if len(v) >= MIN_IMAGES_PER_ID
    }

    print(f"Usable identities: {len(by_id)}")

    # ---------- Build templates ----------
    templates = {}
    for label, embs in by_id.items():
        stack = torch.stack(embs).to(device)
        mean_emb = l2_normalize(stack.mean(dim=0))
        templates[label] = mean_emb


    template_labels = list(templates.keys())
    template_matrix = torch.stack([templates[l] for l in template_labels])
    template_matrix = template_matrix.to(device)

    # ---------- Identification ----------
    top1 = 0
    top5 = 0
    total = 0

    for label, embs in tqdm(by_id.items(), desc="Evaluating"):
        embs = torch.stack(embs).to(device)

        for i in range(len(embs)):
            query = embs[i]

            # Remove query from its own template
            remaining = [e for j, e in enumerate(embs) if j != i]
            if len(remaining) == 0:
                continue

            own_template = l2_normalize(torch.stack(remaining).mean(dim=0))

            # Build comparison matrix
            temp_list = []
            temp_ids = []

            for tid in template_labels:
                if tid == label:
                    temp_list.append(own_template)
                else:
                    temp_list.append(templates[tid])

                temp_ids.append(tid)

            temp_mat = torch.stack(temp_list)

            # Cosine similarity
            sims = torch.matmul(query.unsqueeze(0), temp_mat.T).squeeze(0)

            # Ranking
            topk = torch.topk(sims, k=min(5, len(sims))).indices.tolist()
            ranked_ids = [temp_ids[i] for i in topk]

            if ranked_ids[0] == label:
                top1 += 1
            if label in ranked_ids:
                top5 += 1

            total += 1

    # ---------- Results ----------
    print("\n================ RESULTS ================")
    print(f"Total queries: {total}")
    print(f"Top-1 Accuracy: {100 * top1 / total:.2f}%")
    print(f"Top-5 Accuracy: {100 * top5 / total:.2f}%")
    print("========================================\n")


if __name__ == "__main__":
    main()
