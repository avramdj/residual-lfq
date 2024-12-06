from typing import Sequence, cast

import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from residual_lfq.lfq import LFQ


class ResidualLFQ(nn.Module):
    """Residual Lookup-free Quantization."""

    def __init__(
        self,
        n_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        scale: float,
        scale_progression: Sequence[float] | None = None,
        commit_loss_weight: float = 0.1,
        codebook_loss_weight: float = 0.1,
        entropy_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.scale = scale
        if scale_progression is None:
            self.scale_progression = 2.0 ** -torch.arange(n_codebooks)
        else:
            self.scale_progression = torch.tensor(scale_progression)
        self.commit_loss_weight = commit_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.codebooks = nn.ModuleList(
            [
                LFQ(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    scale=scale,
                    commit_loss_weight=commit_loss_weight,
                    codebook_loss_weight=codebook_loss_weight,
                    entropy_loss_weight=entropy_loss_weight,
                )
                for scale in self.scale_progression
            ]
        )

    def forward(
        self, x: Float[Tensor, "B C"]
    ) -> tuple[Float[Tensor, "B C"], Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        xh = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        residual = x
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_breakdown = {
            "commit_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "codebook_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
            "entropy_loss": torch.tensor(0.0, device=x.device, dtype=x.dtype),
        }
        for codebook in self.codebooks:
            codebook = cast(LFQ, codebook)
            quantized, cb_loss, cb_loss_breakdown = codebook(residual)
            xh = xh + quantized
            residual = residual - quantized
            loss = loss + cb_loss
            loss_breakdown["commit_loss"] = loss_breakdown["commit_loss"] + cb_loss_breakdown["commit_loss"]
            loss_breakdown["codebook_loss"] = (
                loss_breakdown["codebook_loss"] + cb_loss_breakdown["codebook_loss"]
            )
            loss_breakdown["entropy_loss"] = (
                loss_breakdown["entropy_loss"] + cb_loss_breakdown["entropy_loss"]
            )
        return xh, loss, loss_breakdown
