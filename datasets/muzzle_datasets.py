import os
import re
from PIL import Image
from torch.utils.data import Dataset


class MuzzleDataset(Dataset):
    """
    Dataset for cattle muzzle images.

    Assumes directory structure:
    root/
      ├── animal_001/
      │     ├── *_muzzle_*.jpg
      │     ├── other_images.jpg (ignored)
      ├── animal_002/
    """

    def __init__(self, root_dir, transform=None, muzzle_regex=r".*muzzle.*"):
        self.root_dir = root_dir
        self.transform = transform

        # Compile regex once (case-insensitive)
        self.muzzle_pattern = re.compile(muzzle_regex, re.IGNORECASE)

        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        self._build_index()

    def _build_index(self):
        cow_ids = sorted(os.listdir(self.root_dir))

        current_label = 0

        for cow_id in cow_ids:
            cow_path = os.path.join(self.root_dir, cow_id)
            if not os.path.isdir(cow_path):
                continue

            valid_images = []

            for fname in os.listdir(cow_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                if self.muzzle_pattern.match(fname):
                    valid_images.append(fname)

            # Skip identities with <2 images (triplet loss requirement)
            if len(valid_images) < 2:
                continue

            self.label_to_idx[cow_id] = current_label

            for fname in valid_images:
                self.image_paths.append(os.path.join(cow_path, fname))
                self.labels.append(current_label)

            current_label += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
