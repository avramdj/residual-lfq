from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from residual_lfq import ResidualLFQ
import wandb
import os
from PIL import Image
import math

class CelebAAutoencoder(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        n_codebooks: int = 5,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.scale = scale
        codebook_dim = int(math.log2(codebook_size))

        self.encoder = nn.Sequential(
            # 128x128x3 -> 64x64x64
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            # 8x8x512 -> 4x4x1024
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, codebook_dim),
        )

        self.quantizer = ResidualLFQ(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            scale=scale,
            commit_loss_weight=0.02,
            codebook_loss_weight=0.02,
            entropy_loss_weight=0.001,
        )

        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 1024 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (1024, 4, 4)),
            # 4x4x1024 -> 8x8x512
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            # 8x8x512 -> 16x16x256
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            # 32x32x128 -> 64x64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            # 64x64x64 -> 128x128x3
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
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
    model: CelebAAutoencoder,
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

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch}\t"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}\t"  # type: ignore[arg-type]
                    f"({100. * batch_idx / len(train_loader):.0f}%)\t"
                    f"Loss: {loss.item():.6f}\t"
                    f"Recon: {recon_loss.item():.6f}\t"
                    f"Commit: {loss_breakdown['commit_loss'].item():.6f}"
                )
                if batch_idx % 500 == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_recon, _, _, _ = model(data[:4])
                        images = wandb.Image(
                            torch.cat([data[:4], eval_recon], dim=0),
                            caption="Top: Original, Bottom: Reconstructed"
                        )
                        wandb.log({"train/reconstructions": images}, step=step)
                    model.train()


class LocalCelebADataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = self.transform(Image.open(img_name).convert('RGB'))
        return image, 0


def main() -> None:
    wandb.init(
        project="residual-lfq-celeba",
        config={
            "n_codebooks": 5,
            "codebook_size": 16384,  # 2 ** 14
            "scale": 1.0,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "epochs": 100,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("celeba_rlfq_model.pt")

    data_path = Path("data")
    data_path.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])

    train_dataset = LocalCelebADataset(
        root_dir=str(data_path / "img_align_celeba"),
        transform=transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = CelebAAutoencoder(
        codebook_size=wandb.config.codebook_size,
        n_codebooks=wandb.config.n_codebooks,
        scale=wandb.config.scale,
    ).to(device)

    wandb.watch(model, log="all")

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)

    for epoch in range(1, wandb.config.epochs + 1):
        train(model, train_loader, optimizer, device, epoch)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        wandb.save(str(model_path))

    wandb.finish()


if __name__ == "__main__":
    main() 