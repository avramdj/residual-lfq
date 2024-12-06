from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from residual_lfq import ResidualLFQ
import wandb


class CIFARAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        n_codebooks: int = 4,
        codebook_size: int = 2,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # 32x32x3 -> 16x16x32
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            # 16x16x32 -> 8x8x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            # 8x8x64 -> 4x4x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        self.quantizer = ResidualLFQ(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            scale=scale,
            commit_loss_weight=0.1,
            codebook_loss_weight=0.1,
            entropy_loss_weight=0.001,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 4, 4)),
            # 4x4x128 -> 8x8x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            # 8x8x64 -> 16x16x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            # 16x16x32 -> 32x32x3
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encoder(x)
        z_q, loss, loss_breakdown = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, loss, loss_breakdown


def train(
    model: CIFARAutoencoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon, z, quantizer_loss, loss_breakdown = model(data)
        recon_loss = F.mse_loss(recon, data)

        loss = recon_loss + quantizer_loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            metrics = {
                "train/reconstruction_loss": recon_loss.item(),
                "train/quantizer_loss": quantizer_loss.item(),
                "train/total_loss": loss.item(),
                "train/commit_loss": loss_breakdown["commit_loss"].item(),
                "train/codebook_loss": loss_breakdown["codebook_loss"].item(),
                "train/entropy_loss": loss_breakdown["entropy_loss"].item(),
            }
            wandb.log(metrics, step=step)

            if batch_idx % 500 == 0:
                print(
                    f"Train Epoch: {epoch}\t"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}\t"
                    f"({500. * batch_idx / len(train_loader):.0f}%)\t"
                    f"Loss: {loss.item():.6f}\t"
                    f"Recon: {recon_loss.item():.6f}\t"
                    f"Commit: {loss_breakdown['commit_loss'].item():.6f}\t"
                    f"Codebook: {loss_breakdown['codebook_loss'].item():.6f}\t"
                    f"Entropy: {loss_breakdown['entropy_loss'].item():.6f}"
                )

                images = wandb.Image(data[0], caption="Original")
                reconstructed = wandb.Image(recon[0], caption="Reconstructed")
                wandb.log({"train/original": images, "train/reconstructed": reconstructed}, step=step)


def main() -> None:
    wandb.init(
        project="residual-lfq-cifar",
        config={
            "latent_dim": 256,
            "n_codebooks": 4,
            "codebook_size": 4,
            "scale": 1.0,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "epochs": 30,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("data")
    data_path.mkdir(exist_ok=True)

    train_dataset = CIFAR10(root=str(data_path), train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    model = CIFARAutoencoder(
        latent_dim=wandb.config.latent_dim,
        n_codebooks=wandb.config.n_codebooks,
        codebook_size=wandb.config.codebook_size,
        scale=wandb.config.scale,
    ).to(device)

    wandb.watch(model, log="all")

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(1, wandb.config.epochs + 1):
        train(model, train_loader, optimizer, device, epoch)

    model_path = Path("cifar_rlfq_model.pt")
    torch.save(model.state_dict(), model_path)
    wandb.save(str(model_path))

    wandb.finish()


if __name__ == "__main__":
    main()
