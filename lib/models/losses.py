# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
from einops import rearrange, reduce, repeat

import torch as th


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)


class ActLocMSELoss(nn.Module):
    def __init__(self):
        super(ActLocMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        batch_size = target.size(0)
        num_frames = target.size(1)
        num_actions = target.size(2)
        if len(output.size()) == 2:
            output = rearrange(output, "b (t c) -> b t c", t=num_frames, c=num_actions)
        heatmaps_pred = output.reshape((batch_size, num_actions, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_actions, -1)).split(1, 1)
        loss = 0
        for idx in range(num_actions):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_actions


class MaskedStepModelingLoss(nn.Module):
    def __init__(self, mask_ratio):
        super(MaskedStepModelingLoss, self).__init__()
        self.mask_ratio = mask_ratio
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def generate_mask(self, batch_size, num_clips):
        """Returns a binary mask of size (b, m) [batch_size x num_clips]
        where 0 means masked and 1 means unmasked."""
        prob_matrix = th.ones((batch_size, num_clips)) * self.mask_ratio
        bernoulli_samples = th.bernoulli(prob_matrix)
        if self.mask_ratio > 0:
            # Ensure we are masking something, otherwise we would waste
            # an iteration
            while bernoulli_samples.sum() == 0:
                bernoulli_samples = th.bernoulli(prob_matrix)
        mask = 1 - bernoulli_samples
        return mask

    def forward(self, pred_step_logits, step_targets, mask):
        """pred_step_logits: logits of step predictions, size (b, m, s)
        step_targets: pseudo-labels of steps
                      size (b, m, s) -- containing probabilities of steps
        mask: binary mask of size (b, m)
        Returns the average cross entropy classification loss for only the
        masked out steps (where the mask == 0)
        """
        pred_step_logits = rearrange(pred_step_logits, "b m s -> (b m) s")
        step_targets = rearrange(step_targets, "b m s -> (b m) s")
        mask = rearrange(mask, "b m -> (b m)")
        
        all_step_loss = self.criterion(pred_step_logits, step_targets)
        masked_step_loss = all_step_loss * (1 - mask)
        # Average over batch and over clips
        mean_masked_step_loss = masked_step_loss.sum() / (1 - mask).sum()
        return mean_masked_step_loss

class OrderingLoss(nn.Module):
    def __init__(self):
        super(OrderingLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def generate_mask(self, batch_size, num_clips):
        """Returns a binary mask of size (b, m) [batch_size x num_clips]
        where 0 means masked and 1 means unmasked."""
        prob_matrix = th.ones((batch_size, num_clips)) * self.mask_ratio
        bernoulli_samples = th.bernoulli(prob_matrix)
        if self.mask_ratio > 0:
            # Ensure we are masking something, otherwise we would waste
            # an iteration
            while bernoulli_samples.sum() == 0:
                bernoulli_samples = th.bernoulli(prob_matrix)
        mask = 1 - bernoulli_samples
        return mask

    def forward(self, pred_step_logits, step_targets, mask):
        """pred_step_logits: logits of step predictions, size (b, m, s)
        step_targets: pseudo-labels of steps
                      size (b, m, s) -- containing probabilities of steps
        mask: binary mask of size (b, m)
        Returns the average cross entropy classification loss for only the
        masked out steps (where the mask == 0)
        """
        pred_step_logits = rearrange(pred_step_logits, "b m s -> (b m) s")
        step_targets = rearrange(step_targets, "b m s -> (b m) s")
        mask = rearrange(mask, "b m -> (b m)")
        
        all_step_loss = self.criterion(pred_step_logits, step_targets)
        masked_step_loss = all_step_loss * (1 - mask)
        # Average over batch and over clips
        mean_masked_step_loss = masked_step_loss.sum() / (1 - mask).sum()
        return mean_masked_step_loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "milnce": MILNCELoss,
    "MSM": MaskedStepModelingLoss,
    "order": OrderingLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
