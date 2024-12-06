from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from residual_lfq import ResidualLFQ
import wandb


class MNISTAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        n_codebooks: int = 3,
        codebook_size: int = 2,
        scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 14x14 -> 7x7
            nn.Conv2d(16, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, latent_dim),
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
            nn.Linear(latent_dim, 16 * 7 * 7),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (16, 7, 7)),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
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
    model: MNISTAutoencoder,
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

            if batch_idx % 200 == 0:
                print(
                    f"Train Epoch: {epoch}\t"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}\t"
                    f"({200. * batch_idx / len(train_loader):.0f}%)\t"
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
        project="residual-lfq-mnist",
        config={
            "latent_dim": 64,
            "n_codebooks": 5,
            "codebook_size": 16,
            "scale": 1.0,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "epochs": 15,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("data")
    data_path.mkdir(exist_ok=True)

    train_dataset = MNIST(root=str(data_path), train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    model = MNISTAutoencoder(
        latent_dim=wandb.config.latent_dim,
        n_codebooks=wandb.config.n_codebooks,
        codebook_size=wandb.config.codebook_size,
        scale=wandb.config.scale,
    ).to(device)

    wandb.watch(model, log="all")

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(1, wandb.config.epochs + 1):
        train(model, train_loader, optimizer, device, epoch)

    model_path = Path("mnist_rlfq_model.pt")
    torch.save(model.state_dict(), model_path)
    wandb.save(str(model_path))

    wandb.finish()


if __name__ == "__main__":
    main()
