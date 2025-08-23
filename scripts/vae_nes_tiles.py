import re
from typing import List, Tuple

# -----------------------
# Palette and Colorization Utilities
# -----------------------
PALETTE_BLOCKS = {
        "RP2C04-0001": """
755,637,700,447,044,120,222,704,777,333,750,503,403,660,320,777
357,653,310,360,467,657,764,027,760,276,000,200,666,444,707,014
003,567,757,070,077,022,053,507,000,420,747,510,407,006,740,000
000,140,555,031,572,326,770,630,020,036,040,111,773,737,430,473
""",
        "RP2C04-0002": """
000,750,430,572,473,737,044,567,700,407,773,747,777,637,467,040
020,357,510,666,053,360,200,447,222,707,003,276,657,320,000,326
403,764,740,757,036,310,555,006,507,760,333,120,027,000,660,777
653,111,070,630,022,014,704,140,000,077,420,770,755,503,031,444
""",
        "RP2C04-0003": """
507,737,473,555,040,777,567,120,014,000,764,320,704,666,653,467
447,044,503,027,140,430,630,053,333,326,000,006,700,510,747,755
637,020,003,770,111,750,740,777,360,403,357,707,036,444,000,310
077,200,572,757,420,070,660,222,031,000,657,773,407,276,760,022
""",
        "RP2C04-0004": """
430,326,044,660,000,755,014,630,555,310,070,003,764,770,040,572
737,200,027,747,000,222,510,740,653,053,447,140,403,000,473,357
503,031,420,006,407,507,333,704,022,666,036,020,111,773,444,707
757,777,320,700,760,276,777,467,000,750,637,567,360,657,077,120
"""
}

def build_palette(name: str) -> List[Tuple[int, int, int]]:
    def _triplet_to_rgb(code: str) -> Tuple[int, int, int]:
        code = code.strip()
        if not re.fullmatch(r"[0-7]{3}", code):
            return (0, 0, 0)
        r = int(code[0])
        g = int(code[1])
        b = int(code[2])
        def scale(v):
            return int(round((v / 7.0) * 255))
        return (scale(r), scale(g), scale(b))
    if name not in PALETTE_BLOCKS:
        raise ValueError(f"Unknown palette '{name}'.")
    block = PALETTE_BLOCKS[name].strip()
    tokens = [t for t in re.split(r"[,\s]+", block) if t]
    if len(tokens) != 64:
        raise ValueError(f"Palette '{name}' parsed {len(tokens)} entries, expected 64.")
    return [_triplet_to_rgb(t) for t in tokens]

def colorize_tile_from_palette(
    img,
    palette64: List[Tuple[int, int, int]],
    map_indices: List[int] = [0, 1, 2, 3]
) -> 'Image.Image':
    import numpy as np
    from PIL import Image
    arr = np.array(img.convert("L"))
    h, w = arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    c0, c1, c2, c3 = [palette64[i] for i in map_indices]
    lut = {
        0: np.array(c0, dtype=np.uint8),
        1: np.array(c1, dtype=np.uint8),
        2: np.array(c2, dtype=np.uint8),
        3: np.array(c3, dtype=np.uint8),
    }
    for v, col in lut.items():
        mask = (arr == v)
        out[mask] = col
    return Image.fromarray(out, mode="RGB")

def upscale(img, factor: int):
    if factor <= 1:
        return img
    from PIL import Image
    return img.resize((img.width * factor, img.height * factor), resample=Image.NEAREST)
# vae_nes_tiles.py
import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# -----------------------
# Variational Autoencoder
# -----------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=16, conditional=False, num_classes=0, emb_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes

        if conditional:
            self.embedding = nn.Embedding(num_classes, emb_dim)
            enc_in = 1 + emb_dim
        else:
            enc_in = 1

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(enc_in, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc_fc_mu = nn.Linear(32*2*2, latent_dim)
        self.enc_fc_logvar = nn.Linear(32*2*2, latent_dim)

        # Decoder
        dec_in = latent_dim + (emb_dim if conditional else 0)
        self.dec_fc = nn.Linear(dec_in, 32*2*2)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, y=None):
        if self.conditional:
            emb = self.embedding(y).unsqueeze(2).unsqueeze(3)
            emb = emb.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, emb], dim=1)
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.enc_fc_mu(h), self.enc_fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y=None):
        if self.conditional:
            z = torch.cat([z, self.embedding(y)], dim=1)
        h = self.dec_fc(z).view(-1, 32, 2, 2)
        return self.dec_conv(h)

    def forward(self, x, y=None):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# -----------------------
# Loss
# -----------------------
def vae_loss(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="vae_out")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--conditional", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((8, 8)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    num_classes = len(dataset.classes)
    model = VAE(
        latent_dim=args.latent,
        conditional=args.conditional,
        num_classes=num_classes,
        emb_dim=4
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(args.out_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x, y if args.conditional else None)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss {total_loss/len(dataset):.4f}")

        # Save samples
        model.eval()
        with torch.no_grad():
            z = torch.randn(64, args.latent).to(device)
            if args.conditional:
                y_sample = torch.randint(0, num_classes, (64,)).to(device)
                samples = model.decode(z, y_sample)
            else:
                samples = model.decode(z)
            utils.save_image(samples, f"{args.out_dir}/sample_{epoch}.png", nrow=8)

if __name__ == "__main__":
    main()
