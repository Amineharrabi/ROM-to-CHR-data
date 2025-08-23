# sample_vae_tiles_conditional_individual.py
import torch
from vae_nes_tiles import VAE, build_palette, colorize_tile_from_palette, upscale
import argparse
import os
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained VAE .pt file")
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--num_classes", type=int, required=True, help="Number of clusters/classes")
    parser.add_argument("--palette", type=str, default="RP2C04-0004", choices=list(build_palette.keys()))
    parser.add_argument("--palette_map", nargs=4, type=int, default=[0,1,2,3], help="Map tile values 0..3 to palette indices")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--cluster", type=int, default=None, help="If set, generate only this cluster (0-based index)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = VAE(latent_dim=args.latent,
                conditional=True,
                num_classes=args.num_classes,
                emb_dim=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    clusters_to_generate = [args.cluster] if args.cluster is not None else list(range(args.num_classes))
    palette64 = build_palette(args.palette)

    for cluster_id in clusters_to_generate:
        cluster_dir = os.path.join(args.out_dir, f"cluster{cluster_id:02d}")
        os.makedirs(cluster_dir, exist_ok=True)

        z = torch.randn(args.num_samples, args.latent).to(device)
        y_sample = torch.full((args.num_samples,), cluster_id, dtype=torch.long).to(device)
        samples = model.decode(z, y_sample)

        # Convert to 0..3 grayscale
        samples = torch.clamp(samples, 0, 1)
        samples = torch.round(samples * 3).long()

        for i in range(samples.size(0)):
            tile = samples[i,0].cpu().numpy()
            img = Image.fromarray(tile.astype(np.uint8), mode="L")
            img = colorize_tile_from_palette(img, palette64, args.palette_map)
            img = upscale(img, args.upscale)
            tile_path = os.path.join(cluster_dir, f"tile_{i:03d}.png")
            img.save(tile_path)

        print(f"Saved {args.num_samples} tiles to {cluster_dir}")

if __name__ == "__main__":
    main()
