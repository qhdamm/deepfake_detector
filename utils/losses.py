import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)



# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """

#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):

#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

#         return loss_contrastive



class ContrastiveLoss(GenericPairLoss):
    """
    Modified Contrastive loss function in the style of pytorch-metric-learning.
    Automatically handles pair generation and computes positive/negative losses.
    """

    def __init__(self, neg_margin=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["margin"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        """
        Computes the contrastive loss using positive and negative pair distances.
        :param pos_pair_dist: Tensor of distances for positive pairs.
        :param neg_pair_dist: Tensor of distances for negative pairs.
        :param indices_tuple: Indices of positive and negative pairs.
        :return: Dictionary containing positive and negative losses.
        """
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")

        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)

        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        """
        Calculates the loss for a pair (positive or negative).
        :param pair_dists: Tensor of distances for the pair.
        :param pos_or_neg: "pos" or "neg" indicating the type of pair.
        :return: Tensor of per-pair losses.
        """
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        return loss_calc_func(pair_dists)

    def pos_calc(self, pos_pair_dist):
        """Positive pair loss calculation."""
        return torch.mean((1 - 0) * torch.pow(pos_pair_dist, 2))

    def neg_calc(self, neg_pair_dist):
        """Negative pair loss calculation."""
        return torch.mean(torch.pow(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0), 2))

    def forward(self, embeddings, labels):
        """
        Overrides the forward method to calculate pairwise distances and apply the loss.
        :param embeddings: Tensor of shape (batch_size, embedding_dim).
        :param labels: Tensor of shape (batch_size,).
        :return: Contrastive loss.
        """
        pairwise_distances = pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        indices_tuple = lmu.get_all_pairs_indices(labels)

        pos_pair_dist = pairwise_distances[indices_tuple[0], indices_tuple[1]]
        neg_pair_dist = pairwise_distances[indices_tuple[2], indices_tuple[3]]

        losses = self._compute_loss(pos_pair_dist, neg_pair_dist, indices_tuple)
        total_loss = torch.mean(losses["pos_loss"]["losses"] + losses["neg_loss"]["losses"])

        return total_loss

    def get_default_reducer(self):
        """
        Returns the default reducer for averaging non-zero losses.
        """
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        """Defines sub-loss names for logging purposes."""
        return ["pos_loss", "neg_loss"]

    
class MyContrastiveLoss(GenericPairLoss):
    def __init__(self, neg_margin=1.0, recon_weight=0.1, real_weight=0.3, delta=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.recon_weight = recon_weight
        self.real_weight = real_weight
        self.delta = delta
        self.add_to_recordable_attributes(
            list_of_names=["neg_margin", "recon_weight", "delta"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist):
        pos_loss = torch.mean(pos_pair_dist ** 2) if pos_pair_dist.numel() > 0 else 0.0
        neg_loss = torch.mean(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0) ** 2) if neg_pair_dist.numel() > 0 else 0.0
        return pos_loss, neg_loss

    def forward(self, data, labels):
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        # Stack all images and normalize embeddings
        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)
        all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
        embeddings = torch.nn.functional.normalize(all_images, p=2, dim=1)

        # Efficient pairwise distance calculation
        pairwise_distances = 1 - torch.mm(embeddings, embeddings.T)

        # Create positive and negative mask
        labels = labels.reshape(-1)
        real_mask = labels == 1
        fake_mask = labels == 0

        real_indices = real_mask.nonzero(as_tuple=True)[0]
        fake_indices = fake_mask.nonzero(as_tuple=True)[0]

        pos_mask = real_indices.unsqueeze(1) != real_indices.unsqueeze(0)
        pos_pair_dist = pairwise_distances[real_indices][:, real_indices][pos_mask]

        neg_pair_dist = pairwise_distances[real_indices][:, fake_indices].view(-1)

        # Compute contrastive loss
        pos_loss, neg_loss = self._compute_loss(pos_pair_dist, neg_pair_dist)
        contrastive_loss = pos_loss + neg_loss

        # Compute reconstruction loss
        fake_recon_loss = torch.mean((fake_images - fake_recons) ** 2)
        real_recon_loss = torch.mean(
            torch.clamp(self.delta - torch.norm(real_images - real_recons, dim=1), min=0.0) ** 2
        )
        reconstruction_loss = (1 - self.real_weight) * fake_recon_loss + self.real_weight * real_recon_loss

        # Combine all losses
        total_loss = contrastive_loss + self.recon_weight * reconstruction_loss
        return total_loss

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss", "reconstruction_loss"]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt

        return loss


# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_name='ContrastiveLoss', embedding_size=1024, pos_margin=0.0, neg_margin=1.0,
                 memory_size=None, use_miner=False, num_classes=2, tau=0.5, recon_weight=0.1, real_weight=0.3, delta=1.0):
        super(CombinedLoss, self).__init__()
        self.loss_name = loss_name
        self.embedding_size = embedding_size
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.memory_size = memory_size
        self.use_miner = use_miner
        self.num_classes = num_classes
        self.recon_weight = recon_weight
        self.real_weight = real_weight
        self.delta = delta

        if loss_name == 'TripletMarginLoss':
            self.loss_fn = losses.TripletMarginLoss(smooth_loss=True)
        elif loss_name == 'ArcFaceLoss':
            self.loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'SubCenterArcFaceLoss':
            self.loss_fn = losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'CircleLoss':
            self.loss_fn = losses.CircleLoss()
        elif loss_name == 'NTXentLoss':
            self.loss_fn = losses.NTXentLoss(temperature=tau)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif loss_name == 'MyContrastiveLoss':
            self.loss_fn = MyContrastiveLoss(neg_margin=neg_margin, recon_weight=recon_weight, real_weight=real_weight, delta=delta)
        else:
            self.loss_fn = ContrastiveLoss(neg_margin=neg_margin)

        miner = miners.MultiSimilarityMiner() if use_miner else None
        if memory_size is not None:
            self.loss_fn = losses.CrossBatchMemory(self.loss_fn, embedding_size=embedding_size, memory_size=memory_size, miner=miner)

    def forward(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)

        return loss
