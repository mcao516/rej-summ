# -*- coding: utf-8 -*-
"""Implementation of rejection loss

Check our paper: Learning with Rejection for Abstractive Text Summarization

"""
import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

from dataclasses import dataclass, field


def label_smoothed_nll_loss_with_rejection(
    lprobs,
    target,
    epsilon,
    ignore_index=None,
    reduce=True,
    mask=None,
    alpha=1.0,
    unk_idx=3
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # ================== calculate rejection loss ==================
    rej_prob = torch.exp(lprobs[:, unk_idx]).unsqueeze(-1)
    if mask is not None:
        mask = mask.unsqueeze(-1).eq(0)
        keep_prob = (1. - rej_prob).masked_fill(mask, 1.0)  # 0: non-entity
    else:
        keep_prob = 1. - rej_prob
    assert keep_prob.shape == nll_loss.shape, \
        "nll_loss: {}; keep_prob: {}".format(nll_loss.shape, keep_prob.shape)    

    rej_loss = keep_prob * (nll_loss + torch.log(keep_prob))
    rej_regularizer = -alpha * torch.log(keep_prob)
    nll_loss = rej_loss + rej_regularizer

    rej_smooth_loss = keep_prob * (smooth_loss + torch.log(keep_prob))
    smooth_loss = rej_smooth_loss + rej_regularizer
    # ===============================================================

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@dataclass
class LabelSmoothedCrossEntropyCriterionWithRejectionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    rejection_alpha: float = field(
        default=1.0,
        metadata={"help": "weight for the rejection loss regularizer"},
    )


@register_criterion(
    "label_smoothed_cross_entropy_with_rejection",
    dataclass=LabelSmoothedCrossEntropyCriterionWithRejectionConfig,
)
class LabelSmoothedCrossEntropyCriterionWithRejection(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, rejection_alpha):
        super().__init__(task, sentence_avg, label_smoothing)
        self.rejection_alpha = rejection_alpha

        if hasattr(self.task, "target_dictionary"):
            self.unk_idx = self.task.target_dictionary.unk()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        # This mask marks all entities in the summary sequence. If the mask is not None,
        # rejection loss only applies to entity tokens.
        mask = None
        if "mask" in sample and sample["mask"] is not None:
            mask = sample["mask"].view(-1)
            assert target.size() == mask.size(), \
                "Target size: {}; Mask size: {}.".format(target.size(), mask.size())

        loss, nll_loss = label_smoothed_nll_loss_with_rejection(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            mask=mask,
            alpha=self.rejection_alpha,
            unk_idx=self.unk_idx,
        )
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
