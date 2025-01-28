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

    
# class MyContrastiveLoss(GenericPairLoss):
#     def __init__(self, neg_margin=1.0, recon_weight=0.1, real_weight=0.3, delta=1.0, **kwargs):
#         super().__init__(mat_based_loss=False, **kwargs)
#         self.neg_margin = neg_margin
#         self.recon_weight = recon_weight
#         self.real_weight = real_weight
#         self.delta = delta
#         self.add_to_recordable_attributes(
#             list_of_names=["neg_margin", "recon_weight", "delta"], is_stat=False
#         )

#     def _compute_loss(self, pos_pair_dist, neg_pair_dist):
#         pos_loss = torch.mean(pos_pair_dist ** 2) if pos_pair_dist.numel() > 0 else 0.0
#         neg_loss = torch.mean(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0) ** 2) if neg_pair_dist.numel() > 0 else 0.0
#         return pos_loss, neg_loss

#     def forward(self, data, labels):
#         real_images, real_recons, fake_images, fake_recons = zip(*data)

#         # Stack all images and normalize embeddings
#         real_images = torch.stack(real_images)
#         real_recons = torch.stack(real_recons)
#         fake_images = torch.stack(fake_images)
#         fake_recons = torch.stack(fake_recons)
#         all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
#         embeddings = torch.nn.functional.normalize(all_images, p=2, dim=1)

#         # Efficient pairwise distance calculation
#         pairwise_distances = 1 - torch.mm(embeddings, embeddings.T)

#         # Create positive and negative mask
#         labels = labels.reshape(-1)
#         real_mask = labels == 1
#         fake_mask = labels == 0

#         real_indices = real_mask.nonzero(as_tuple=True)[0]
#         fake_indices = fake_mask.nonzero(as_tuple=True)[0]

#         pos_mask = real_indices.unsqueeze(1) != real_indices.unsqueeze(0)
#         pos_pair_dist = pairwise_distances[real_indices][:, real_indices][pos_mask]

#         neg_pair_dist = pairwise_distances[real_indices][:, fake_indices].view(-1)

#         # Compute contrastive loss
#         pos_loss, neg_loss = self._compute_loss(pos_pair_dist, neg_pair_dist)
#         contrastive_loss = pos_loss + neg_loss

#         # Compute reconstruction loss
#         fake_recon_loss = torch.mean((fake_images - fake_recons) ** 2)
#         real_recon_loss = torch.mean(
#             torch.clamp(self.delta - torch.norm(real_images - real_recons, dim=1), min=0.0) ** 2
#         )
#         reconstruction_loss = (1 - self.real_weight) * fake_recon_loss + self.real_weight * real_recon_loss

#         # Combine all losses
#         total_loss = contrastive_loss + self.recon_weight * reconstruction_loss
#         return total_loss

#     def get_default_reducer(self):
#         return AvgNonZeroReducer()

#     def _sub_loss_names(self):
#         return ["pos_loss", "neg_loss", "reconstruction_loss"]
    
import torch
import torch.nn as nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from torch.nn.functional import cosine_similarity

import torch
import torch.nn.functional as F

class MyContrastiveLoss1(GenericPairLoss):
    def __init__(self, neg_margin=1.0, real_weight=0.5, recon_weight=0.1, delta=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.real_weight = real_weight
        self.recon_weight = recon_weight
        self.delta = delta
        self.add_to_recordable_attributes(
            list_of_names=["neg_margin", "recon_weight", "delta"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist):
        # For positive pairs, minimize cosine distance (close to 0)
        pos_loss = torch.mean(pos_pair_dist ** 2) if pos_pair_dist.numel() > 0 else 0.0

        # For negative pairs, enforce margin (distance >= neg_margin)
        neg_loss = torch.mean(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0) ** 2) if neg_pair_dist.numel() > 0 else 0.0

        return pos_loss, neg_loss


    def forward(self, data, labels):
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)

        # Normalize embeddings for cosine similarity
        all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
        embeddings = F.normalize(all_images, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        cosine_similarities = torch.mm(embeddings, embeddings.T)
        cosine_distances = 1 - cosine_similarities

        # Create masks for real and fake
        labels = labels.reshape(-1)
        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        fake_indices = (labels == 0).nonzero(as_tuple=True)[0]

        # Positive pairs: real-real similarity
        pos_pair_sim = cosine_distances[real_indices][:, real_indices]
        pos_pair_sim = pos_pair_sim[torch.triu_indices(len(real_indices), len(real_indices), offset=1)]

        # Negative pairs: real-fake similarity
        neg_pair_sim = cosine_distances[real_indices][:, fake_indices].view(-1)

        # Compute contrastive loss
        pos_loss, neg_loss = self._compute_loss(pos_pair_sim, neg_pair_sim)
        contrastive_loss = pos_loss + neg_loss

        # Compute reconstruction loss
        fake_recon_loss = torch.mean((fake_images - fake_recons) ** 2)
        real_recon_loss = torch.mean(
            torch.clamp(self.delta - torch.norm(real_images - real_recons, dim=1), min=0.0) ** 2
        )
        reconstruction_loss = (1 - self.real_weight) * fake_recon_loss + self.real_weight * real_recon_loss
        total_loss = contrastive_loss + self.recon_weight * reconstruction_loss

        return total_loss
    



class MyContrastiveLoss2(GenericPairLoss):
    def __init__(self, neg_margin=1.0, recon_weight=0.1, delta=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.recon_weight = recon_weight
        self.delta = delta
        self.add_to_recordable_attributes(
            list_of_names=["neg_margin", "recon_weight", "delta"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist):
        # For positive pairs, minimize cosine distance (close to 0)
        pos_loss = torch.mean(pos_pair_dist ** 2) if pos_pair_dist.numel() > 0 else 0.0

        # For negative pairs, enforce margin (distance >= neg_margin)
        neg_loss = torch.mean(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0) ** 2) if neg_pair_dist.numel() > 0 else 0.0

        return pos_loss, neg_loss

    def forward(self, data, labels):
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)

        # Normalize embeddings for cosine similarity
        all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
        embeddings = F.normalize(all_images, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        cosine_similarities = torch.mm(embeddings, embeddings.T)
        cosine_distances = 1 - cosine_similarities  

        # Create masks for real and fake
        labels = labels.reshape(-1)
        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        fake_indices = (labels == 0).nonzero(as_tuple=True)[0]

        # Positive pairs: real-real similarity
        pos_pair_sim = cosine_distances[real_indices][:, real_indices]
        pos_pair_sim = pos_pair_sim[torch.triu_indices(len(real_indices), len(real_indices), offset=1)]

        # Negative pairs: real-fake similarity
        neg_pair_sim = cosine_distances[real_indices][:, fake_indices].view(-1)

        # Compute contrastive loss
        pos_loss, neg_loss = self._compute_loss(pos_pair_sim, neg_pair_sim)
        contrastive_loss = pos_loss + neg_loss

        # Compute reconstruction difference loss
        real_recon_dist = torch.norm(real_images - real_recons, dim=1, p=2) ** 2
        fake_recon_dist = torch.norm(fake_images - fake_recons, dim=1, p=2) ** 2
        # real_recon_dist = torch.norm(real_images - real_recons, p=1, dim=1)
        # fake_recon_dist = torch.norm(fake_images - fake_recons, p=1, dim=1) 


        recon_diff_loss = torch.mean(torch.clamp(real_recon_dist - fake_recon_dist + self.delta, min=0.0))

        # Combine all losses
        total_loss = contrastive_loss + self.recon_weight * recon_diff_loss
        return total_loss
    
import torch
import torch.nn as nn

class MyContrastiveLoss3(nn.Module):
    """
    Custom loss function that focuses only on the reconstruction loss component.
    It calculates the reconstruction loss for both real and fake images.
    """
    def __init__(self, recon_weight=0.5, delta=1.5):
        super(MyContrastiveLoss3, self).__init__()
        self.recon_weight = recon_weight  # Weight for real vs fake reconstruction loss
        self.delta = delta  # Margin for real image reconstruction
        self.add_to_recordable_attributes = ["recon_weight", "delta"]

    def reconstruction_loss(self, real, real_recon, fake, fake_recon):
        """
        Computes the weighted reconstruction loss.
        
        :param real: Tensor of real images.
        :param real_recon: Tensor of reconstructed real images.
        :param fake: Tensor of fake images.
        :param fake_recon: Tensor of reconstructed fake images.
        :return: Reconstruction loss.
        """
        # Compute reconstruction loss for real and fake samples
        real_recon_loss = torch.mean(torch.clamp(self.delta - torch.norm(real - real_recon, dim=1, p=2), min=0.0) ** 2)
        fake_recon_loss = torch.mean(torch.norm(fake - fake_recon, dim=1, p=2) ** 2)

        # Weighted combination of real and fake reconstruction loss
        return (1 - self.recon_weight) * fake_recon_loss + self.recon_weight * real_recon_loss

    def forward(self, data, labels):
        """
        Forward pass to compute the reconstruction loss.
        
        :param real: Tensor of real images.
        :param real_recon: Tensor of reconstructed real images.
        :param fake: Tensor of fake images.
        :param fake_recon: Tensor of reconstructed fake images.
        :return: Total reconstruction loss.
        """
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)
        recon_loss = self.reconstruction_loss(real_images, real_recons, fake_images, fake_recons)
        return recon_loss



import torch
import torch.nn.functional as F
import scipy.linalg

class MyContrastiveLossEU(GenericPairLoss):
    def __init__(self, neg_margin=1.0, recon_weight=0.1, delta=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.recon_weight = recon_weight
        self.delta = delta
        self.add_to_recordable_attributes(
            list_of_names=["neg_margin", "recon_weight", "delta"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist):
        # For positive pairs, minimize Euclidean distance (close to 0)
        pos_loss = torch.mean(pos_pair_dist ** 2) if pos_pair_dist.numel() > 0 else 0.0
        
        # For negative pairs, enforce margin separation
        neg_loss = torch.mean(torch.clamp(self.neg_margin - neg_pair_dist, min=0.0) ** 2) if neg_pair_dist.numel() > 0 else 0.0
        return pos_loss, neg_loss

    def forward(self, data, labels):
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)

        # Flatten the images to treat them as feature vectors for distance computation
        all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
        all_images_flattened = all_images.view(all_images.size(0), -1)

        # Compute pairwise Euclidean distances
        dist_matrix = torch.cdist(all_images_flattened, all_images_flattened, p=2)

        # Create masks for real and fake
        labels = labels.reshape(-1)
        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        fake_indices = (labels == 0).nonzero(as_tuple=True)[0]

        # Positive pairs: real-real distance
        pos_pair_dist = dist_matrix[real_indices][:, real_indices]
        pos_pair_dist = pos_pair_dist[torch.triu_indices(len(real_indices), len(real_indices), offset=1)]

        # Negative pairs: real-fake distance
        neg_pair_dist = dist_matrix[real_indices][:, fake_indices].view(-1)

        # Compute contrastive loss using Euclidean distance
        pos_loss, neg_loss = self._compute_loss(pos_pair_dist, neg_pair_dist)
        contrastive_loss = pos_loss + neg_loss

        # Compute reconstruction difference loss (L1 distance)
        real_recon_dist = torch.norm(real_images - real_recons, p=1, dim=1)
        fake_recon_dist = torch.norm(fake_images - fake_recons, p=1, dim=1) 

        recon_diff_loss = torch.mean(torch.clamp(real_recon_dist - fake_recon_dist + self.delta, min=0.0))

        # Combine all losses
        total_loss = contrastive_loss + self.recon_weight * recon_diff_loss
        return total_loss


class MyContrastiveLossFID(GenericPairLoss):
    def __init__(self, neg_margin=1.0, recon_weight=0.1, delta=1.0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.recon_weight = recon_weight
        self.delta = delta
        self.add_to_recordable_attributes(
            list_of_names=["recon_weight", "delta"], is_stat=False
        )

    def compute_fid(self, real_embeddings, fake_embeddings):
        # Compute means and covariances
        real_mean = real_embeddings.mean(dim=0)
        fake_mean = fake_embeddings.mean(dim=0)

        real_cov = torch.cov(real_embeddings.T)
        fake_cov = torch.cov(fake_embeddings.T)

        # Convert to NumPy for sqrtm calculation
        real_cov_np = real_cov.cpu().detach().numpy()
        fake_cov_np = fake_cov.cpu().detach().numpy()

        # Compute sqrtm using scipy
        cov_sqrt, _ = scipy.linalg.sqrtm(real_cov_np @ fake_cov_np, disp=False)

        # Handle numerical instability (convert back to tensor and take real part)
        cov_sqrt = torch.tensor(cov_sqrt.real, dtype=real_cov.dtype, device=real_cov.device)

        # Compute FID score
        mean_diff = torch.norm(real_mean - fake_mean, p=2) ** 2
        fid_score = mean_diff + torch.trace(real_cov + fake_cov - 2 * cov_sqrt)

        return fid_score

    def forward(self, data, labels):
        real_images, real_recons, fake_images, fake_recons = zip(*data)

        real_images = torch.stack(real_images)
        real_recons = torch.stack(real_recons)
        fake_images = torch.stack(fake_images)
        fake_recons = torch.stack(fake_recons)

        # Normalize embeddings
        all_images = torch.cat([real_images, real_recons, fake_images, fake_recons], dim=0)
        embeddings = F.normalize(all_images, p=2, dim=1)

        # Separate real and fake embeddings based on labels
        labels = labels.reshape(-1)
        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        fake_indices = (labels == 0).nonzero(as_tuple=True)[0]

        real_embeddings = embeddings[real_indices]
        fake_embeddings = embeddings[fake_indices]

        # Compute FID score
        fid_loss = self.compute_fid(real_embeddings, fake_embeddings)

        # Compute reconstruction difference loss
        real_recon_dist = torch.norm(real_images - real_recons, p=1, dim=1)
        fake_recon_dist = torch.norm(fake_images - fake_recons, p=1, dim=1)
        recon_diff_loss = torch.mean(torch.clamp(real_recon_dist - fake_recon_dist + self.delta, min=0.0))

        # Combine all losses
        total_loss = fid_loss + self.recon_weight * recon_diff_loss
        return total_loss




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
        elif loss_name == 'ContrastiveLoss':
            self.loss_fn = ContrastiveLoss(neg_margin=neg_margin)
        elif loss_name == 'MyContrastiveLoss1':
            self.loss_fn = MyContrastiveLoss1(neg_margin=neg_margin, recon_weight=recon_weight, delta=delta)
        elif loss_name == 'MyContrastiveLoss2':
            self.loss_fn = MyContrastiveLoss2(neg_margin=neg_margin, recon_weight=recon_weight, delta=delta)
        elif loss_name == 'MyContrastiveLoss3':
            self.loss_fn = MyContrastiveLoss3(recon_weight=recon_weight, delta=delta)
        elif loss_name == 'MyContrastiveLossFID':
            self.loss_fn = MyContrastiveLossFID(neg_margin=neg_margin, recon_weight=recon_weight, delta=delta)
        elif loss_name == 'MyContrastiveLossEU':
            self.loss_fn = MyContrastiveLossEU(neg_margin=neg_margin, recon_weight=recon_weight, delta=delta)
        else:
            self.loss_fn = ContrastiveLoss(neg_margin=neg_margin)

        miner = miners.MultiSimilarityMiner() if use_miner else None
        if memory_size is not None:
            self.loss_fn = losses.CrossBatchMemory(self.loss_fn, embedding_size=embedding_size, memory_size=memory_size, miner=miner)

    def forward(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)

        return loss
