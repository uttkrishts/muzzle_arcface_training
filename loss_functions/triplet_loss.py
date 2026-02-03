from pytorch_metric_learning import losses, miners, distances


class TripletLossWithMining:
    """
    Triplet loss with semi-hard negative mining
    using cosine distance on L2-normalized embeddings.
    """

    def __init__(self, margin=0.2):
        # Distance function
        self.distance = distances.CosineSimilarity()

        # Semi-hard triplet miner
        self.miner = miners.TripletMarginMiner(
            margin=margin,
            type_of_triplets="semi-hard",
            distance=self.distance
        )

        # Triplet loss
        self.loss_fn = losses.TripletMarginLoss(
            margin=margin,
            distance=self.distance
        )

    def __call__(self, embeddings, labels):
        """
        embeddings: (B, D) L2-normalized
        labels: (B,)
        """
        triplets = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, triplets)
        return loss
