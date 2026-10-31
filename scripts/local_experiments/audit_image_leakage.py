"""Audit a class-folder image dataset for duplicates and likely leakage.

Reports exact file/pixel duplicates, perceptual near-duplicates (including
90-degree rotations and centre crops), conflicting labels, and available EXIF
metadata. A patient identity cannot be inferred reliably without patient IDs.
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fftpack import dct

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return np.packbits((pixels[:, 1:] > pixels[:, :-1]).reshape(-1)).tobytes()


def phash(image):
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coeff = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    values = coeff.reshape(-1)[1:]
    return np.packbits(values > np.median(values)).tobytes()


def variants(image):
    """Original, rotations, and a centre crop for modest crop/rotation changes."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    margin_x, margin_y = int(width * 0.08), int(height * 0.08)
    crop = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    return [image, image.rotate(90, expand=True), image.rotate(180, expand=True),
            image.rotate(270, expand=True), crop]


def hamming(left, right):
    return sum((a ^ b).bit_count() for a, b in zip(left, right))


def scan(root):
    records = []
    for split in ("train", "val", "test"):
        folder = root / split
        if not folder.is_dir():
            continue
        for label_dir in sorted(folder.iterdir()):
            if not label_dir.is_dir():
                continue
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    exif = source.getexif()
                    pixel_digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hashes = [phash(item) for item in variants(image)]
                    records.append({
                        "path": str(path.resolve()), "split": split, "label": label_dir.name,
                        "width": image.width, "height": image.height, "file_sha256": file_digest,
                        "pixel_sha256": pixel_digest, "dhash": dhash(image).hex(),
                        "phash": hashes[0].hex(), "variant_hashes": [item.hex() for item in hashes],
                        "exif_present": bool(exif),
                        "exif_datetime": str(exif.get(306, "")),
                        "exif_make": str(exif.get(271, "")),
                        "exif_model": str(exif.get(272, "")),
                    })
    return records


def find_near_duplicates(records, threshold):
    # Index 16-bit blocks of each pHash variant. This keeps comparisons tractable;
    # final candidate confirmation uses the full 63-bit perceptual hash.
    buckets = defaultdict(set)
    for index, record in enumerate(records):
        for variant in record["variant_hashes"]:
            raw = bytes.fromhex(variant)
            for offset in range(0, 8, 2):
                buckets[(offset, raw[offset:offset + 2])].add(index)
    pairs = set()
    for values in buckets.values():
        values = sorted(values)
        for position, left in enumerate(values):
            for right in values[position + 1:]:
                pairs.add((left, right))

    output = []
    for left, right in sorted(pairs):
        a, b = records[left], records[right]
        minimum = min(hamming(bytes.fromhex(x), bytes.fromhex(y))
                      for x in a["variant_hashes"] for y in b["variant_hashes"])
        if minimum <= threshold:
            output.append({
                "path_a": a["path"], "split_a": a["split"], "label_a": a["label"],
                "path_b": b["path"], "split_b": b["split"], "label_b": b["label"],
                "phash_distance": minimum,
                "same_label": a["label"] == b["label"],
                "cross_split": a["split"] != b["split"],
            })
    return pd.DataFrame(output)


def grouped_duplicates(frame, column):
    groups = frame.groupby(column).filter(lambda item: len(item) > 1).sort_values(column)
    return groups.drop(columns=["variant_hashes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("image_audit"))
    parser.add_argument("--near-threshold", type=int, default=8,
                        help="0 = identical pHash; 8 is a conservative near-duplicate threshold.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    records = scan(args.data_root)
    frame = pd.DataFrame(records)
    frame.drop(columns=["variant_hashes"]).to_csv(args.output / "image_manifest.csv", index=False)
    exact_files = grouped_duplicates(frame, "file_sha256")
    exact_pixels = grouped_duplicates(frame, "pixel_sha256")
    exact_files.to_csv(args.output / "exact_file_duplicates.csv", index=False)
    exact_pixels.to_csv(args.output / "exact_pixel_duplicates.csv", index=False)
    near = find_near_duplicates(records, args.near_threshold)
    near.to_csv(args.output / "near_duplicates.csv", index=False)
    conflicts = near[near["same_label"] == False] if len(near) else near
    conflicts.to_csv(args.output / "near_duplicates_conflicting_labels.csv", index=False)
    metadata = frame[frame["exif_present"]].drop(columns=["variant_hashes"])
    metadata.to_csv(args.output / "images_with_exif.csv", index=False)

    print(f"Images scanned: {len(frame)}")
    print(f"Exact file-duplicate rows: {len(exact_files)}")
    print(f"Exact decoded-pixel duplicate rows: {len(exact_pixels)}")
    print(f"Near-duplicate pairs (pHash distance <= {args.near_threshold}): {len(near)}")
    print(f"Near-duplicate pairs with conflicting labels: {len(conflicts)}")
    print(f"Images with EXIF metadata: {len(metadata)}")
    print(f"Reports: {args.output.resolve()}")


if __name__ == "__main__":
    main()
