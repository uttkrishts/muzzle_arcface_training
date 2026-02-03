import random
from collections import defaultdict
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    P x K sampler for metric learning.

    Each batch contains:
      - P unique labels
      - K samples per label
    """

    def __init__(self, labels, P, K):
        self.labels = labels
        self.P = P
        self.K = K

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Remove labels with insufficient samples
        self.label_to_indices = {
            label: idxs
            for label, idxs in self.label_to_indices.items()
            if len(idxs) >= K
        }

        self.valid_labels = list(self.label_to_indices.keys())

        if len(self.valid_labels) < P:
            raise ValueError(
                f"Not enough labels ({len(self.valid_labels)}) "
                f"to sample P={P} identities"
            )

    def __iter__(self):
        random.shuffle(self.valid_labels)

        batch = []

        for label in self.valid_labels:
            indices = self.label_to_indices[label]
            selected = random.sample(indices, self.K)
            batch.extend(selected)

            if len(batch) == self.P * self.K:
                yield from batch
