import pytest
import torch
from residual_lfq.rlfq import ResidualLFQ


@pytest.fixture
def rlfq_model() -> ResidualLFQ:
    return ResidualLFQ(
        n_codebooks=4,
        codebook_size=256,
        codebook_dim=8,
        scale=1.0,
        scale_progression=None,
        commit_loss_weight=0.1,
        codebook_loss_weight=0.1,
        entropy_loss_weight=1.0,
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(4, 8)  # batch_size=4, codebook_dim=8


def test_rlfq_initialization(rlfq_model: ResidualLFQ) -> None:
    assert rlfq_model.n_codebooks == 4
    assert rlfq_model.codebook_size == 256
    assert rlfq_model.codebook_dim == 8
    assert rlfq_model.scale == 1.0
    assert len(rlfq_model.codebooks) == 4

    # Test scale progression
    expected_scales = 2.0 ** -torch.arange(4)
    assert torch.allclose(rlfq_model.scale_progression, expected_scales)


def test_rlfq_custom_scale_progression() -> None:
    custom_scales = [1.0, 0.5, 0.25, 0.125]
    model = ResidualLFQ(
        n_codebooks=4,
        codebook_size=256,
        codebook_dim=8,
        scale=1.0,
        scale_progression=custom_scales,
    )
    assert torch.allclose(model.scale_progression, torch.tensor(custom_scales))


def test_forward(rlfq_model: ResidualLFQ, sample_input: torch.Tensor) -> None:
    quantized, total_loss, loss_breakdown = rlfq_model(sample_input)

    assert quantized.shape == sample_input.shape
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0

    expected_keys = {"commit_loss", "codebook_loss", "entropy_loss"}
    assert set(loss_breakdown.keys()) == expected_keys

    # Check if total loss equals sum of individual losses
    computed_total = sum(loss_breakdown.values())
    assert torch.allclose(total_loss, computed_total)


def test_residual_reduction(rlfq_model: ResidualLFQ, sample_input: torch.Tensor) -> None:
    # Test if the residual gets smaller with each codebook
    residual = sample_input.clone()
    for codebook in rlfq_model.codebooks:
        quantized, _, _ = codebook(residual)
        new_residual = residual - quantized

        # Check if the new residual has smaller magnitude
        assert torch.mean(torch.abs(new_residual)) <= torch.mean(torch.abs(residual))
        residual = new_residual


def test_zero_loss_weights() -> None:
    model = ResidualLFQ(
        n_codebooks=4,
        codebook_size=256,
        codebook_dim=8,
        scale=1.0,
        commit_loss_weight=0.0,
        codebook_loss_weight=0.0,
        entropy_loss_weight=0.0,
    )

    x = torch.randn(4, 8)
    _, total_loss, loss_breakdown = model(x)

    assert total_loss == 0.0
    assert all(v == 0.0 for v in loss_breakdown.values())


def test_rlfq_to_index(rlfq_model: ResidualLFQ, sample_input: torch.Tensor) -> None:
    # Get quantized values from forward pass
    quantized, _, _ = rlfq_model(sample_input)
    
    # Convert to indices
    indices = rlfq_model.to_index(quantized)
    
    # Check shape and type
    assert indices.shape == (sample_input.shape[0], rlfq_model.n_codebooks)
    assert indices.dtype == torch.long
    
    # Check index ranges
    assert (indices >= 0).all()
    assert (indices < 2**rlfq_model.codebook_dim).all()


def test_rlfq_from_index(rlfq_model: ResidualLFQ, sample_input: torch.Tensor) -> None:
    # Get quantized values and convert to indices
    quantized, _, _ = rlfq_model(sample_input)
    indices = rlfq_model.to_index(quantized)
    
    # Reconstruct from indices
    reconstructed = rlfq_model.from_index(indices)
    
    # Check shape
    assert reconstructed.shape == sample_input.shape
    
    # Check reconstruction accuracy
    assert torch.allclose(reconstructed, quantized)


def test_rlfq_index_conversion_pipeline(rlfq_model: ResidualLFQ, sample_input: torch.Tensor) -> None:
    """Test the full pipeline: input -> quantized -> indices -> reconstructed."""
    # Forward pass
    quantized, _, _ = rlfq_model(sample_input)
    
    # Convert to indices and back
    indices = rlfq_model.to_index(quantized)
    reconstructed = rlfq_model.from_index(indices)
    
    # Check if we can perfectly reconstruct the quantized values
    assert torch.allclose(reconstructed, quantized)
    
    # Check if indices have expected properties
    assert indices.shape == (sample_input.shape[0], rlfq_model.n_codebooks)
    assert indices.dtype == torch.long
    assert (indices >= 0).all()
    assert (indices < 2**rlfq_model.codebook_dim).all()
