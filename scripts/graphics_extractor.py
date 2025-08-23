import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
from scripts.RomParse import parse_nes_header

# NES 4-color palette (basic)
NES_PALETTE = [
    (84, 84, 84), 
    (0, 30, 116), 
    (8, 16, 144), 
    (48, 0, 136)
]

def extract_chr_data(path):
    """Extract raw CHR (graphics) data from an NES ROM."""
    header = parse_nes_header(path)
    with open(path, "rb") as f:
        f.seek(16 + header["prg_size"])  # skip header + PRG
        chr_data = f.read(header["chr_size"])
    if len(chr_data) == 0:
        raise ValueError("No CHR data found in this ROM.")
    return chr_data

def chr_to_tiles(chr_data):
    """Convert raw CHR data into a list of 8x8 tiles (NumPy arrays)."""
    tiles = []
    for i in range(0, len(chr_data)-15, 16):  # each tile is 16 bytes
        tile = np.zeros((8, 8), dtype=np.uint8)
        for row in range(8):
            plane0 = chr_data[i + row]
            plane1 = chr_data[i + row + 8]
            for col in range(8):
                bit0 = (plane0 >> (7 - col)) & 1
                bit1 = (plane1 >> (7 - col)) & 1
                tile[row, col] = (bit1 << 1) | bit0
        tiles.append(tile)
    return tiles

def tiles_to_images(tiles, palette=NES_PALETTE):
    """Convert list of 8x8 tiles (NumPy arrays) into PIL images."""
    imgs = []
    for t in tiles:
        img = Image.new("RGB", (8, 8))
        pixels = img.load()
        for y in range(8):
            for x in range(8):
                pixels[x, y] = palette[t[y, x]]
        imgs.append(img)
    return imgs

# --- Quick test ---
if __name__ == "__main__":
    test_rom = "roms/kongJapan.nes"
    chr_data = extract_chr_data(test_rom)
    tiles = chr_to_tiles(chr_data)
    images = tiles_to_images(tiles)
    print(f"Total tiles extracted: {len(images)}")
    if len(images) > 0:
        images[0].show()
