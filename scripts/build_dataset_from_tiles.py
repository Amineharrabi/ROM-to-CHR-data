"""
Build a PyTorch-ready ImageFolder dataset from extracted NES tiles (PNG).





Features:
- Optional colorization of grayscale tiles (0..3) using RP2C04-0001..0004 palettes
- KMeans clustering to group similar tiles (reduces duplicates)
- Exports ImageFolder layout: dataset/cluster00/, cluster01/, ...
- Optional "unique_tiles" export by taking the cluster centroids' nearest members
- Optional upscaling for visualization


Usage (examples):
    python build_dataset_from_tiles.py --in_dir tiles/ --out_dir dataset/ --clusters 60 --colorize --palette RP2C04-0004 --palette-map 0 1 2 3 --upscale 4
    python build_dataset_from_tiles.py --in_dir tiles/ --out_dir dataset/ --clusters 40

Requires:
    pip install pillow numpy scikit-learn
(optional)  pip install tqdm
"""

import argparse
import os
import re
import shutil
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


# -------------------------
# PALETTE PARSING UTILITIES
# -------------------------

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

def _triplet_to_rgb(code: str) -> Tuple[int, int, int]:
    """
    Convert a 3-digit string like '755' to RGB on 0..255 scale,
    interpreting each digit (0..7) as intensity /7.
    """
    code = code.strip()
    if not re.fullmatch(r"[0-7]{3}", code):
        # If weird token (blank), default to black
        return (0, 0, 0)
    r = int(code[0])
    g = int(code[1])
    b = int(code[2])
    def scale(v):  # map 0..7 -> 0..255
        return int(round((v / 7.0) * 255))
    return (scale(r), scale(g), scale(b))

def build_palette(name: str) -> List[Tuple[int, int, int]]:
    """
    Return a 64-color palette list for RP2C04-000X using the pasted blocks.
    """
    if name not in PALETTE_BLOCKS:
        raise ValueError(f"Unknown palette '{name}'. Choose one of: {list(PALETTE_BLOCKS.keys())}")
    block = PALETTE_BLOCKS[name].strip()
    tokens = [t for t in re.split(r"[,\s]+", block) if t]
    if len(tokens) != 64:
        raise ValueError(f"Palette '{name}' parsed {len(tokens)} entries, expected 64.")
    return [_triplet_to_rgb(t) for t in tokens]


# -------------------------
# IMAGE / TILE HELPERS
# -------------------------

def is_grayscale_0_3(img: Image.Image) -> bool:
    """
    Detect whether an image is 0..3 grayscale (common for raw CHR tiles).
    """
    arr = np.array(img.convert("L"))
    uniques = np.unique(arr)
    return np.all(np.isin(uniques, [0, 1, 2, 3]))

def colorize_tile_from_palette(
    img: Image.Image,
    palette64: List[Tuple[int, int, int]],
    map_indices: List[int] = [0, 1, 2, 3]
) -> Image.Image:
    """
    Map pixel values 0..3 to 4 chosen entries of a 64-color palette.
    'map_indices' picks the four palette entries used for values 0..3.
    """
    if len(map_indices) != 4 or any(not (0 <= i < 64) for i in map_indices):
        raise ValueError("map_indices must be 4 ints each in range 0..63")
    arr = np.array(img.convert("L"))  # (H,W) values 0..3 expected
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

def upscale(img: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return img
    # NEAREST preserves crisp pixels
    return img.resize((img.width * factor, img.height * factor), resample=Image.NEAREST)


# -------------------------
# CLUSTERING PIPELINE
# -------------------------

def load_tiles_as_vectors(
    in_dir: str,
    use_grayscale_for_clustering: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Load all PNGs from in_dir and return:
      - X: N x D array (flattened pixels)
      - files: list of file paths (same order as X)
    Clustering is done on grayscale by default for robustness.
    """
    files = []
    for fname in sorted(os.listdir(in_dir)):
        if fname.lower().endswith(".png"):
            files.append(os.path.join(in_dir, fname))
    if not files:
        raise RuntimeError(f"No PNG files found in '{in_dir}'")

    vecs = []
    valid_files = []
    for path in (tqdm(files, desc="Loading tiles") if TQDM else files):
        try:
            img = Image.open(path)
            if use_grayscale_for_clustering:
                arr = np.array(img.convert("L"))  # HxW
            else:
                arr = np.array(img.convert("RGB"))  # HxWx3
            vecs.append(arr.flatten().astype(np.float32))
            valid_files.append(path)
        except Exception as e:
            print(f"Skipping unreadable image: {path} ({e})")
    if not vecs:
        raise RuntimeError("No valid PNG files found for clustering.")
    X = np.stack(vecs, axis=0)  # N x D
    # Normalize features to 0..1
    if X.max() > 0:
        X /= X.max()
    return X, valid_files

def run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 0) -> Tuple[np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def pick_representatives_per_cluster(
    X: np.ndarray, labels: np.ndarray, kmeans: KMeans, files: List[str]
) -> Dict[int, str]:
    """
    For each cluster, pick the tile closest to the centroid (as representative).
    """
    reps = {}
    centroids = kmeans.cluster_centers_
    for c in range(centroids.shape[0]):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        # Euclidean distance to centroid
        dists = np.linalg.norm(X[idxs] - centroids[c], axis=1)
        best_local = idxs[np.argmin(dists)]
        reps[c] = files[best_local]
    return reps


# -------------------------
# EXPORT
# -------------------------

def export_imagefolder(
    files: List[str],
    labels: np.ndarray,
    out_dir: str,
    colorize: bool,
    palette_name: str,
    map_indices: List[int],
    upscale_factor: int,
    also_export_unique: bool,
    use_original_if_colored: bool = True
):
    """
    Create dataset/clusterXX/… structure.
    If colorize=True and tile is 0..3 grayscale, colorize using chosen palette.
    If tile already colored and use_original_if_colored=True, keep original colors.
    """
    os.makedirs(out_dir, exist_ok=True)
    palette64 = build_palette(palette_name) if colorize else None

    # cluster folders
    for c in np.unique(labels):
        os.makedirs(os.path.join(out_dir, f"cluster{c:02d}"), exist_ok=True)

    # copy/transform tiles
    iterator = zip(files, labels)
    if TQDM:
        iterator = tqdm(iterator, total=len(files), desc="Exporting ImageFolder")
    for path, lab in iterator:
        img = Image.open(path)
        # colorize if requested and grayscale 0..3
        if colorize and is_grayscale_0_3(img):
            img = colorize_tile_from_palette(img, palette64, map_indices)
        elif colorize and not is_grayscale_0_3(img) and not use_original_if_colored:
            # force colorization by quantizing to 0..3 first (optional branch)
            arr = np.array(img.convert("L"))
            arr = np.clip(np.round(arr / (arr.max() if arr.max() else 1) * 3), 0, 3).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
            img = colorize_tile_from_palette(img, palette64, map_indices)

        img = upscale(img, upscale_factor)
        out_path = os.path.join(out_dir, f"cluster{lab:02d}", os.path.basename(path))
        img.save(out_path)

    if also_export_unique:
        # Create a flat folder of "one representative per cluster"
        uniq_dir = os.path.join(out_dir, "_unique_tiles")
        os.makedirs(uniq_dir, exist_ok=True)
        # Note: representatives are created by a separate call that passes reps dict
        # This function expects caller to copy them in after computing reps.
        pass  # handled in main()


# -------------------------
# MAIN (CLI)
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing extracted tile PNGs")
    ap.add_argument("--out_dir", required=True, help="Output dataset folder (ImageFolder layout)")
    ap.add_argument("--clusters", type=int, default=50, help="Number of KMeans clusters")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for KMeans")
    ap.add_argument("--colorize", action="store_true", help="Apply RP2C04 palette to 0..3 grayscale tiles")
    ap.add_argument("--palette", default="RP2C04-0004", choices=list(PALETTE_BLOCKS.keys()),
                    help="Which RP2C04 palette variant to use if --colorize")
    ap.add_argument("--palette-map", nargs=4, type=int, default=[0, 1, 2, 3],
                    help="Map tile values 0..3 to these 64-color indices (four ints 0..63)")
    ap.add_argument("--upscale", type=int, default=1, help="Integer upscale factor for saved images (1=no upscale)")
    ap.add_argument("--unique", action="store_true", help="Also export one representative tile per cluster")
    ap.add_argument("--force-gray-cluster", action="store_true",
                    help="Cluster on grayscale even if images are colored (recommended)")
    args = ap.parse_args()

    # 1) Load as vectors for clustering
    X, files = load_tiles_as_vectors(args.in_dir, use_grayscale_for_clustering=True or args.force_gray_cluster)

    # 2) KMeans
    labels, kmeans = run_kmeans(X, n_clusters=args.clusters, seed=args.seed)

    # 3) Export ImageFolder (and optionally unique tiles)
    export_imagefolder(
        files=files,
        labels=labels,
        out_dir=args.out_dir,
        colorize=args.colorize,
        palette_name=args.palette,
        map_indices=args.palette_map,
        upscale_factor=args.upscale,
        also_export_unique=args.unique
    )

    if args.unique:
        reps = pick_representatives_per_cluster(X, labels, kmeans, files)
        uniq_dir = os.path.join(args.out_dir, "_unique_tiles")
        for c, path in (tqdm(reps.items(), desc="Exporting unique tiles") if TQDM else reps.items()):
            img = Image.open(path)
            if args.colorize and is_grayscale_0_3(img):
                img = colorize_tile_from_palette(img, build_palette(args.palette), args.palette_map)
            img = upscale(img, args.upscale)
            img.save(os.path.join(uniq_dir, f"cluster{c:02d}_" + os.path.basename(path)))

    print("\nDone.")
    print(f"- ImageFolder dataset at: {args.out_dir}")
    if args.unique:
        print(f"- Representatives in: {os.path.join(args.out_dir, '_unique_tiles')}")


if __name__ == "__main__":
    main()
