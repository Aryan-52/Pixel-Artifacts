"""Source-attribution head: which generator most likely made an AI image.

Input is the fp-v4.1 standardised five-pillar feature vector (the same features the detector
reads, concatenated in PILLARS order). Output is a softmax over the generators in the
attribution training set. It is trained on a separate labelled dataset (Tiny-GenImage plus ProGAN
training images), never on the zero-shot test set, and it does not change the detector's verdict.
"""
from __future__ import annotations

import torch
from torch import nn

PILLARS = ["spatial", "frequency", "color", "physical", "semantic"]


class AttributionHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def concat(feats: dict) -> torch.Tensor:
    return torch.cat([feats[k].float() for k in PILLARS], dim=1)


def save(path, model: AttributionHead, classes: list[str], unknown_below: float, meta: dict):
    torch.save({"state": model.state_dict(), "in_dim": model.net[0].in_features,
                "hidden": model.net[0].out_features, "classes": classes,
                "unknown_below": unknown_below, "meta": meta}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = AttributionHead(ck["in_dim"], len(ck["classes"]), ck["hidden"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck
