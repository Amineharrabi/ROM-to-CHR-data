import os , pandas as pd
import numpy as np
from PIL import Image

def save_dataset(rom , tiles , out_dir='dataset'):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    NES_PALETTE = [
        (84, 84, 84),
        (0, 30, 116),
        (8, 16, 144),
        (48, 0, 136)
    ]
    for idx, t in enumerate(tiles):
        if t.shape != (8, 8):
            continue  # skip invalid tiles
        img = Image.new("RGB", (8, 8))
        pixels = img.load()
        for y in range(8):
            for x in range(8):
                pixels[x, y] = NES_PALETTE[t[y, x]]
        path = f"{out_dir}/{rom}_tile{idx}.png"
        img.save(path)
        meta.append({
            "rom": rom,
            "tile_index": idx,
            "image_path": path
        })
    df = pd.DataFrame(meta)
    df.to_csv(f"{out_dir}/{rom}_dataset.csv", index=False)

