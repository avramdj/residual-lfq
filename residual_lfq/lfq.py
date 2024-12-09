from typing import cast
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class LFQ(nn.Module):
    """Lookup-free quantization."""

    def __init__(
        self,
        codebook_size: int | None = None,
        codebook_dim: int | None = None,
        scale: float = 1.0,
        commit_loss_weight: float = 0.1,
        codebook_loss_weight: float = 0.1,
        entropy_loss_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if codebook_size is not None and codebook_dim is not None:
            raise ValueError("Either codebook_size or codebook_dim must be provided, not both")
        if codebook_size is not None:
            self.codebook_size = codebook_size
            self.codebook_dim = int(math.log2(codebook_size))
        elif codebook_dim is not None:
            self.codebook_size = 2 ** codebook_dim
            self.codebook_dim = codebook_dim
        else:
            raise ValueError("Either codebook_size or codebook_dim must be provided")
        self.scale = scale
        self.commit_loss_weight = commit_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.max_entropy = float(torch.log2(torch.tensor([self.codebook_dim])))
        self.register_buffer("exp2mask", 2 ** torch.arange(self.codebook_dim, dtype=torch.float32))

    def quantize(self, x: Float[Tensor, "B C"]) -> Float[Tensor, "B C"]:
        x = torch.tanh(x)
        z = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        quantized = z * self.scale

        if self.training:
            return x * self.scale + (quantized - x * self.scale).detach()
        return quantized

    def commit_loss(self, x: Float[Tensor, "B C"], quantized: Float[Tensor, "B C"]) -> Float[Tensor, ""]:
        if self.commit_loss_weight <= 0:
            return torch.tensor(0.0, device=x.device)
        return F.mse_loss(x, quantized.detach(), reduction="mean") * self.commit_loss_weight

    def codebook_loss(self, x: Float[Tensor, "B C"], quantized: Float[Tensor, "B C"]) -> Float[Tensor, ""]:
        if self.codebook_loss_weight <= 0:
            return torch.tensor(0.0, device=x.device)
        return F.mse_loss(x.detach(), quantized, reduction="mean") * self.codebook_loss_weight

    def entropy_loss(self, x: Float[Tensor, "B C"]) -> Float[Tensor, ""]:
        if self.entropy_loss_weight <= 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        eps = 1e-8
        x_norm = x / self.scale
        pos_prob = torch.mean((x_norm > 0).float(), dim=0)
        neg_prob = 1 - pos_prob
        pos_prob = torch.clamp(pos_prob, eps, 1 - eps)
        neg_prob = torch.clamp(neg_prob, eps, 1 - eps)
        entropy_per_dim = -(pos_prob * torch.log2(pos_prob) + neg_prob * torch.log2(neg_prob))
        avg_entropy = torch.mean(entropy_per_dim)
        max_entropy = 1.0
        entropy_loss = (
            torch.clamp((max_entropy - avg_entropy) / max_entropy, 0.0, 1.0) * self.entropy_loss_weight
        )

        return entropy_loss

    def forward(
        self, x: Float[Tensor, "B C"]
    ) -> tuple[Float[Tensor, "B C"], Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        quantized = self.quantize(x)
        commit_loss = self.commit_loss(x, quantized)
        codebook_loss = self.codebook_loss(x, quantized)
        entropy_loss = self.entropy_loss(quantized)

        total_loss = commit_loss + codebook_loss + entropy_loss

        loss_breakdown = {
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "entropy_loss": entropy_loss,
        }
        return quantized, total_loss, loss_breakdown

    def to_index(self, quantized: Float[Tensor, "B C"]) -> Float[Tensor, " B"]:
        bits = (quantized > 0).float()
        return cast(Float[Tensor, " B"], (bits * self.exp2mask).sum(dim=1).long())

    def from_index(self, index: Float[Tensor, " B"]) -> Float[Tensor, "B C"]:
        bits = cast(Float[Tensor, " B C"], ((index.unsqueeze(1) & self.exp2mask.long()) != 0).float())
        quantized = torch.where(bits > 0, self.scale, -self.scale)
        return quantized
