import pytest
import torch
from residual_lfq.lfq import LFQ


@pytest.fixture
def lfq_model() -> LFQ:
    return LFQ(
        codebook_size=256,
        codebook_dim=8,
        scale=1.0,
        commit_loss_weight=0.1,
        codebook_loss_weight=0.1,
        entropy_loss_weight=1.0,
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(4, 8)  # batch_size=4, codebook_dim=8


def test_lfq_initialization(lfq_model: LFQ) -> None:
    assert lfq_model.codebook_size == 256
    assert lfq_model.codebook_dim == 8
    assert lfq_model.scale == 1.0
    assert lfq_model.commit_loss_weight == 0.1
    assert lfq_model.codebook_loss_weight == 0.1
    assert lfq_model.entropy_loss_weight == 1.0


def test_quantize(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    assert quantized.shape == sample_input.shape
    # Check if quantized values are either +scale or -scale
    unique_values = torch.unique(torch.abs(quantized))
    assert len(unique_values) == 1
    assert torch.allclose(unique_values[0], torch.tensor(lfq_model.scale))


def test_commit_loss(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    loss = lfq_model.commit_loss(sample_input, quantized)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0  # loss should be non-negative


def test_codebook_loss(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    loss = lfq_model.codebook_loss(sample_input, quantized)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0


def test_entropy_loss(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    loss = lfq_model.entropy_loss(quantized)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert 0 <= loss <= lfq_model.entropy_loss_weight


def test_forward(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized, total_loss, loss_breakdown = lfq_model(sample_input)

    assert quantized.shape == sample_input.shape
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0

    expected_keys = {"commit_loss", "codebook_loss", "entropy_loss"}
    assert set(loss_breakdown.keys()) == expected_keys

    # Check if total loss equals sum of individual losses
    computed_total = sum(loss_breakdown.values())
    assert torch.allclose(total_loss, computed_total)


def test_to_index(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    indices = lfq_model.to_index(quantized)

    assert indices.shape == (sample_input.shape[0],)
    assert indices.dtype == torch.long
    assert (indices >= 0).all()
    assert (indices < 2**lfq_model.codebook_dim).all()


def test_from_index(lfq_model: LFQ, sample_input: torch.Tensor) -> None:
    quantized = lfq_model.quantize(sample_input)
    indices = lfq_model.to_index(quantized)
    reconstructed = lfq_model.from_index(indices)

    assert reconstructed.shape == sample_input.shape
    assert torch.allclose(reconstructed, quantized)
