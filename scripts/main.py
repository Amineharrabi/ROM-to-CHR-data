
import os
from graphics_extractor import extract_chr_data, chr_to_tiles
from DataSetBuilder import save_dataset

roms_dir = "roms"
out_dir = "tiles"
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(roms_dir):
    if fname.lower().endswith(".nes"):
        rom_path = os.path.join(roms_dir, fname)
        try:
            chr_data = extract_chr_data(rom_path)
            tiles = chr_to_tiles(chr_data)
            save_dataset(fname.replace('.nes',''), tiles, out_dir=out_dir)
            print(f"Saved tiles for {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")