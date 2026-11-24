from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "valid", "test")
UNLABELED = "Unlabeled"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_COLORS = {"train": "#3b6ea5", "valid": "#e08a3c", "test": "#4c9f70"}
CLASS_COLORS = {
    "Bumblefoot": "#8a5a44",
    "Fowlpox": "#b5523b",
    "Healthy": "#4c9f70",
    "coryza": "#d9a441",
    "crd": "#5c7cba",
    "Unlabeled": "#9aa0a6",
}


@dataclass
class ImageRecord:
    path: str
    split: str
    label: str
    width: int
    height: int
    mode: str
    file_size: int
    brightness: float
    contrast: float
    sharpness: float
    mean_r: float
    mean_g: float
    mean_b: float
    md5: str
    dhash_identity: int
    dhash_hflip: int
    dhash_rot90: int
    dhash_rot180: int
    dhash_rot270: int
    dhash_vflip: int
    dhash_diag: int
    dhash_antidiag: int
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reports/week1"))
    parser.add_argument("--dup-threshold", type=int, default=8,
                        help="Max Hamming distance (of 64 bits) for near-duplicates")
    parser.add_argument("--sample-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_images(data_root: Path) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((path, split, label_dir.name))
    return items


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_from_array(gray: np.ndarray) -> int:
    """64-bit difference hash; robust to small brightness changes."""
    image = Image.fromarray(gray.astype(np.uint8), mode="L")
    small = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def dhash_variants(gray: np.ndarray) -> tuple[int, ...]:
    """Eight dihedral orientations, so flipped/rotated augmentations match."""
    variants = (
        gray,
        np.fliplr(gray),
        np.rot90(gray, 1),
        np.rot90(gray, 2),
        np.rot90(gray, 3),
        np.flipud(gray),
        np.fliplr(np.rot90(gray, 1)),
        np.fliplr(np.rot90(gray, 3)),
    )
    return tuple(dhash_from_array(v) for v in variants)


def laplacian_variance(gray: np.ndarray) -> float:
    interior = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(interior[1:-1, 1:-1].var())


def analyze_image(path: Path, split: str, label: str, root: Path) -> ImageRecord:
    record = ImageRecord(
        path=str(path.relative_to(root)), split=split, label=label,
        width=0, height=0, mode="", file_size=path.stat().st_size,
        brightness=math.nan, contrast=math.nan, sharpness=math.nan,
        mean_r=math.nan, mean_g=math.nan, mean_b=math.nan, md5="",
        dhash_identity=0, dhash_hflip=0, dhash_rot90=0, dhash_rot180=0,
        dhash_rot270=0, dhash_vflip=0, dhash_diag=0, dhash_antidiag=0,
    )
    try:
        record.md5 = file_md5(path)
        with Image.open(path) as handle:
            handle.load()
            record.width, record.height = handle.size
            record.mode = handle.mode
            rgb = handle.convert("RGB")
            gray_image = rgb.convert("L")
            rgb_array = np.asarray(rgb, dtype=np.float32)
            gray = np.asarray(gray_image, dtype=np.float32)
        means = rgb_array.reshape(-1, 3).mean(axis=0)
        record.mean_r, record.mean_g, record.mean_b = (float(v) for v in means)
        record.brightness = float(gray.mean())
        record.contrast = float(gray.std())
        record.sharpness = laplacian_variance(gray)
        hashes = dhash_variants(gray)
        (
            record.dhash_identity, record.dhash_hflip, record.dhash_rot90,
            record.dhash_rot180, record.dhash_rot270, record.dhash_vflip,
            record.dhash_diag, record.dhash_antidiag,
        ) = hashes
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    return record


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def find_near_duplicates(inv: pd.DataFrame, threshold: int) -> pd.DataFrame:
    valid = inv[inv["error"] == ""].reset_index(drop=True)
    hash_columns = [
        "dhash_identity", "dhash_hflip", "dhash_rot90", "dhash_rot180",
        "dhash_rot270", "dhash_vflip", "dhash_diag", "dhash_antidiag",
    ]
    hashes = valid[hash_columns].to_numpy(dtype=np.uint64)
    n = len(valid)
    left_idx: list[np.ndarray] = []
    right_idx: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # min Hamming distance across the 8 orientation variants
        delta = np.bitwise_count(hashes[start:end, None, :] ^ hashes[None, :, :])
        dist = delta.min(axis=2)
        rows, cols = np.where(dist <= threshold)
        global_rows = rows + start
        keep = global_rows < cols
        left_idx.append(global_rows[keep])
        right_idx.append(cols[keep])
        distances.append(dist[rows, cols][keep])

    if not left_idx:
        return pd.DataFrame()
    left = np.concatenate(left_idx)
    right = np.concatenate(right_idx)
    dist = np.concatenate(distances)
    pairs = pd.DataFrame({
        "path_a": valid.loc[left, "path"].to_numpy(),
        "path_b": valid.loc[right, "path"].to_numpy(),
        "split_a": valid.loc[left, "split"].to_numpy(),
        "split_b": valid.loc[right, "split"].to_numpy(),
        "label_a": valid.loc[left, "label"].to_numpy(),
        "label_b": valid.loc[right, "label"].to_numpy(),
        "distance": dist,
    })
    pairs["same_split"] = pairs["split_a"] == pairs["split_b"]
    pairs["same_label"] = pairs["label_a"] == pairs["label_b"]
    pairs["cross_split"] = ~pairs["same_split"]
    return pairs.sort_values("distance").reset_index(drop=True)


def build_clusters(inv: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    valid = inv[inv["error"] == ""].reset_index(drop=True)
    index_of = {p: i for i, p in enumerate(valid["path"])}
    uf = UnionFind(len(valid))
    for a, b in zip(pairs["path_a"], pairs["path_b"]):
        uf.union(index_of[a], index_of[b])

    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(len(valid)):
        grouped[uf.find(i)].append(i)

    rows = []
    for root, members in grouped.items():
        if len(members) < 2:
            continue
        subset = valid.iloc[members]
        rows.append({
            "cluster_id": root,
            "size": len(members),
            "splits": "|".join(sorted(subset["split"].unique())),
            "n_splits": int(subset["split"].nunique()),
            "labels": "|".join(sorted(subset["label"].unique())),
            "mixed_labels": bool(subset["label"].nunique() > 1),
            "paths": " | ".join(subset["path"]),
        })
    clusters = pd.DataFrame(rows)
    if not clusters.empty:
        clusters = clusters.sort_values(["n_splits", "size"], ascending=False).reset_index(drop=True)
    return clusters


def leakage_table(inv: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return pd.DataFrame(columns=["split", "label", "images_in_cross_split_clusters",
                                     "split_total", "leakage_pct"])
    cross_paths: set[str] = set()
    for row in clusters[clusters["n_splits"] > 1].itertuples():
        cross_paths.update(p.strip() for p in row.paths.split(" | "))
    inv = inv.copy()
    inv["cross_split_dup"] = inv["path"].isin(cross_paths)
    rows = []
    for split in SPLITS:
        split_df = inv[inv["split"] == split]
        for label in sorted(split_df["label"].unique()):
            sub = split_df[split_df["label"] == label]
            leaked = int(sub["cross_split_dup"].sum())
            rows.append({
                "split": split,
                "label": label,
                "images_in_cross_split_clusters": leaked,
                "split_total": len(sub),
                "leakage_pct": 100.0 * leaked / len(sub) if len(sub) else 0.0,
            })
    return pd.DataFrame(rows)


def plot_class_distribution(inv: pd.DataFrame, labels: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for split in SPLITS:
        counts = np.array([
            int(((inv["split"] == split) & (inv["label"] == label)).sum()) for label in labels
        ])
        ax.barh(y, counts, left=left, color=SPLIT_COLORS[split], label=split, edgecolor="white")
        left += counts
    for i, label in enumerate(labels):
        ax.text(left[i] + max(left) * 0.01, i, f"{int(left[i]):,}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Images")
    ax.set_title("Images per class and split")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(left) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_class_proportions(inv: pd.DataFrame, labels: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(SPLITS))
    bottom = np.zeros(len(SPLITS))
    for label in labels:
        fractions = []
        for split in SPLITS:
            subset = inv[inv["split"] == split]
            fractions.append(100.0 * (subset["label"] == label).mean())
        fractions = np.array(fractions)
        ax.bar(x, fractions, bottom=bottom, label=label,
               color=CLASS_COLORS.get(label, "#777777"), edgecolor="white")
        bottom += fractions
    ax.set_xticks(x, [f"{s}\n(n={int((inv['split'] == s).sum()):,})" for s in SPLITS])
    ax.set_ylabel("% of split")
    ax.set_title("Class composition of each split")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_quality(inv: pd.DataFrame, labels: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, column, title, log_scale in (
        (axes[0], "brightness", "Brightness (mean grayscale)", False),
        (axes[1], "sharpness", "Sharpness (Laplacian variance)", True),
    ):
        data = [inv.loc[inv["label"] == label, column].dropna().to_numpy() for label in labels]
        box = ax.boxplot(data, orientation="vertical", patch_artist=True, widths=0.6,
                         medianprops={"color": "#222222", "linewidth": 1.4},
                         flierprops={"markersize": 2, "alpha": 0.4})
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(CLASS_COLORS.get(label, "#999999"))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Image-quality distribution by class (all splits)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_color_profile(quality: pd.DataFrame, labels: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    subset = quality.set_index("label").reindex(labels)
    x = np.arange(len(labels))
    width = 0.26
    for offset, (column, name, color) in enumerate((
        ("mean_r", "R", "#c0504d"),
        ("mean_g", "G", "#4f9d69"),
        ("mean_b", "B", "#4a6fa5"),
    )):
        ax.bar(x + (offset - 1) * width, subset[column], width, label=name, color=color)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean channel value (0-255)")
    ax.set_title("Color profile by class")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_sample_grid(inv: pd.DataFrame, labels: list[str], root: Path,
                     per_class: int, seed: int, out: Path) -> None:
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(labels), per_class,
                             figsize=(2.1 * per_class, 2.1 * len(labels)))
    axes = np.atleast_2d(axes)
    for row, label in enumerate(labels):
        candidates = inv[(inv["label"] == label) & (inv["error"] == "")]
        paths = candidates["path"].tolist()
        rng.shuffle(paths)
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                with Image.open(root / paths[col]) as image:
                    ax.imshow(np.asarray(image.convert("RGB")))
                ax.set_title(label, fontsize=10, pad=3)
    fig.suptitle("Sample images per class", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_leakage(leakage: pd.DataFrame, out: Path) -> None:
    rates = []
    for split in ("valid", "test"):
        subset = leakage[leakage["split"] == split]
        total = int(subset["split_total"].sum())
        leaked = int(subset["images_in_cross_split_clusters"].sum())
        rates.append((split, leaked, total, 100.0 * leaked / total if total else 0.0))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = [r[0] for r in rates]
    values = [r[3] for r in rates]
    bars = ax.bar(names, values, color=[SPLIT_COLORS["valid"], SPLIT_COLORS["test"]], width=0.55)
    for bar, (_, leaked, total, pct) in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{pct:.1f}%\n({leaked}/{total:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of images with a cross-split near-duplicate")
    ax.set_ylim(0, max(values + [1]) * 1.35)
    ax.set_title("Candidate cross-split leakage (orientation-aware)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_duplicate_examples(pairs: pd.DataFrame, root: Path, out: Path) -> None:
    cross = pairs[pairs["cross_split"]].head(6)
    if cross.empty:
        return
    fig, axes = plt.subplots(len(cross), 2, figsize=(9, 2.5 * len(cross)), squeeze=False)
    for row, pair in enumerate(cross.itertuples()):
        for col, (path, split, label) in enumerate((
            (pair.path_a, pair.split_a, pair.label_a),
            (pair.path_b, pair.split_b, pair.label_b),
        )):
            ax = axes[row, col]
            with Image.open(root / path) as image:
                ax.imshow(np.asarray(image.convert("RGB")))
            ax.set_title(f"{split} · {label}", fontsize=9)
            ax.axis("off")
        axes[row, 0].text(-0.12, 0.5, f"d={pair.distance}", transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Closest cross-split near-duplicate pairs", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_image_sizes(inv: pd.DataFrame, out: Path) -> None:
    sizes = (inv.assign(size=inv["width"].astype(str) + "×" + inv["height"].astype(str))
             .groupby("size").size().sort_values(ascending=False).head(8))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(sizes.index, sizes.to_numpy(), color="#3b6ea5")
    for i, value in enumerate(sizes.to_numpy()):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title("Image dimensions")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def wide_distribution(inv: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        row = {"label": label}
        for split in SPLITS:
            count = int(((inv["split"] == split) & (inv["label"] == label)).sum())
            row[split] = count
        row["total"] = sum(row[s] for s in SPLITS)
        rows.append(row)
    df = pd.DataFrame(rows)
    for split in SPLITS:
        total = int((inv["split"] == split).sum())
        df[f"{split}_pct"] = (100.0 * df[split] / total).round(1) if total else 0.0
    return df


def md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(
            f"{v:.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
            for v in row
        ) + " |")
    return "\n".join(lines)


def write_data_card(inv: pd.DataFrame, dist: pd.DataFrame, labels: list[str],
                    root: Path, out: Path) -> None:
    totals = {split: int((inv["split"] == split).sum()) for split in SPLITS}
    unlabeled = inv[inv["label"] == UNLABELED]
    sizes = inv.groupby(["width", "height"]).size().reset_index(name="count")
    size_text = ", ".join(f"{int(r.width)}×{int(r.height)} ({int(r.count):,})"
                          for r in sizes.itertuples())
    text = f"""# Data Card — Poultry Disease Detection (v9)

## Identity
- **Name:** Poultry Disease Detection v9
- **Source:** Roboflow Universe, project `poultry-disease-detection-quprj`
- **Export date:** 2024-09-18
- **License:** CC BY 4.0 (as stated in `README.dataset.txt`)
- **Task:** single-label image classification of poultry conditions
- **Label source:** folder names in the export; no clinical definitions or expert
  adjudication are included in the dataset package

## Composition
- **Images:** {len(inv):,} total — train {totals['train']:,}, valid {totals['valid']:,}, test {totals['test']:,}
- **Supervised classes:** {len([l for l in labels if l != UNLABELED])} ({", ".join(l for l in labels if l != UNLABELED)})
- **Excluded bucket:** `{UNLABELED}` ({len(unlabeled)} images) — no target label, so it is not used for training or evaluation
- **Dimensions:** {size_text}
- **Formats/modes:** {", ".join(f"{m} ({c:,})" for m, c in inv["mode"].value_counts().items())}
- **Preprocessing (per export README):** auto-orientation, EXIF stripped, stretched to 220×220
- **Augmentation (per export README):** 3 versions per source image — 50% horizontal flip,
  50% vertical flip, 90° rotation, ±15° rotation, ±15% brightness, ±10% exposure

## Intended use
Research/educational prototype for image-based poultry condition triage. It is a
decision-support experiment, not a veterinary diagnostic device.

## Out of scope
- Clinical diagnosis, treatment decisions, or regulatory use
- Birds, conditions, housing systems, or camera setups not represented in this dataset
- Any claim of performance on real farm imagery or on video

## Known limitations
- **Severe class imbalance:** see `class_distribution.csv`; rare classes have few examples.
- **Unaudited labels:** the export does not document who labelled the images or how.
- **Leakage risk:** augmentation produced three versions of each source image, and the
  split may separate those versions. `near_duplicates.csv` quantifies candidate leakage.
- **No metadata:** no bird identity, age, breed, farm, geography, or capture date, so a
  random split cannot measure generalization to new farms.
- **Resolution ceiling:** stretched 220×220 images may erase small lesions.
- **Background shortcut risk:** lighting, bedding, and camera differences may correlate
  with labels; Grad-CAM review is required in Week 4.

## Maintenance
Re-run `python src/eda.py` after any dataset change and keep the generated
`summary_metrics.json` with the released model.
"""
    out.write_text(text)


def write_summary(inv: pd.DataFrame, dist: pd.DataFrame, quality: pd.DataFrame,
                  pairs: pd.DataFrame, clusters: pd.DataFrame, leakage: pd.DataFrame,
                  corrupt: pd.DataFrame, dup_threshold: int, out: Path) -> None:
    totals = {split: int((inv["split"] == split).sum()) for split in SPLITS}
    supervised = [l for l in dist["label"] if l != UNLABELED]
    train_counts = dist.set_index("label")["train"]
    train_counts = train_counts.reindex(supervised)
    largest_label = train_counts.idxmax()
    smallest_label = train_counts.idxmin()
    imbalance = float(train_counts.max() / max(train_counts.min(), 1))
    unlabeled_counts = {split: int(((inv["split"] == split) & (inv["label"] == UNLABELED)).sum())
                        for split in SPLITS}
    exact_groups = inv[inv["error"] == ""].groupby("md5").size()
    exact_groups = exact_groups[exact_groups > 1]
    exact_images = int(exact_groups.sum()) if len(exact_groups) else 0
    cross_pairs = int(pairs["cross_split"].sum()) if not pairs.empty else 0
    cross_clusters = int((clusters["n_splits"] > 1).sum()) if not clusters.empty else 0
    mixed_label_clusters = int(clusters["mixed_labels"].sum()) if not clusters.empty else 0
    leak_test = leakage[leakage["split"] == "test"]
    leak_test_images = int(leak_test["images_in_cross_split_clusters"].sum())
    leak_test_pct = 100.0 * leak_test_images / totals["test"] if totals["test"] else 0.0
    leak_valid_images = int(leakage.loc[leakage["split"] == "valid",
                                        "images_in_cross_split_clusters"].sum())
    leak_valid_pct = 100.0 * leak_valid_images / totals["valid"] if totals["valid"] else 0.0
    size_counts = inv.groupby(["width", "height"]).size()
    unique_sizes = len(size_counts)
    size_text = ", ".join(f"{w}×{h} ({c:,})" for (w, h), c in size_counts.items())

    quality_view = quality[["label", "n", "brightness_median", "contrast_median",
                            "sharpness_median"]].copy()
    quality_view.columns = ["Class", "n", "Median brightness", "Median contrast",
                            "Median sharpness"]

    dist_view = dist[["label", "train", "valid", "test", "total"]].copy()
    dist_view.columns = ["Class", "Train", "Valid", "Test", "Total"]

    top_pairs = (pairs[pairs["cross_split"]].head(5)[["path_a", "path_b", "distance"]]
                 if not pairs.empty else pd.DataFrame())

    text = f"""# Week 1 — EDA Report

*Auto-generated by `src/eda.py`. Figures in `figures/`.*

## 1. Snapshot

| Item | Value |
| --- | --- |
| Total images | {len(inv):,} |
| Train / Valid / Test | {totals['train']:,} / {totals['valid']:,} / {totals['test']:,} |
| Supervised classes | {len(supervised)} |
| Unlabeled images (excluded) | {sum(unlabeled_counts.values())} (train {unlabeled_counts['train']}, valid {unlabeled_counts['valid']}, test {unlabeled_counts['test']}) |
| Image dimensions | {unique_sizes} unique — {size_text} |
| Corrupt / unreadable files | {len(corrupt)} |
| Exact duplicate groups | {len(exact_groups)} groups, {exact_images} images |
| Near-duplicate pairs (≤{dup_threshold}/64 bits) | {len(pairs):,} |
| Cross-split near-duplicate pairs | {cross_pairs:,} |
| Cross-split duplicate clusters | {cross_clusters:,} |
| Clusters with conflicting labels | {mixed_label_clusters:,} |

## 2. Class distribution

{md_table(dist_view)}

![Class distribution](figures/class_distribution.png)

- **Imbalance (train, supervised):** largest `{largest_label}` = {int(train_counts.max()):,};
  smallest `{smallest_label}` = {int(train_counts.min()):,}; **{imbalance:.1f}× ratio**.
- `{UNLABELED}` appears only as an unlabelled bucket and is excluded from all supervised work.
- The augmentation pipeline produced three versions of each source image, so effective
  diversity is lower than the raw counts suggest.

![Split composition](figures/split_composition.png)

## 3. Image quality

{md_table(quality_view)}

![Quality by class](figures/quality_by_class.png)
![Color profile](figures/color_profile.png)

Read this as a **confound check**, not a diagnosis: large brightness/contrast/color gaps
between classes can let a model separate classes using capture conditions rather than
lesions. If the gaps are large, Week 3 must include aggressive color/lighting augmentation
and Week 4 must include Grad-CAM review.

## 4. Duplicates and leakage

- Exact duplicate groups (MD5): **{len(exact_groups)}** covering **{exact_images}** images.
- Orientation-aware near-duplicates: **{len(pairs):,} pairs** at Hamming distance ≤ {dup_threshold}.
- Cross-split pairs: **{cross_pairs:,}**, forming **{cross_clusters:,}** clusters.
- Valid images in cross-split clusters: **{leak_valid_images} / {totals['valid']:,} ({leak_valid_pct:.1f}%)**.
- Test images in cross-split clusters: **{leak_test_images} / {totals['test']:,} ({leak_test_pct:.1f}%)**.
- Clusters containing more than one label: **{mixed_label_clusters}** (either label noise or
  loose near-duplicate matching — inspect these manually).

![Leakage](figures/leakage.png)
![Duplicate examples](figures/duplicate_examples.png)

**Policy decision for Week 2:** treat cross-split clusters as *candidate leakage*. Remove the
**train-side** members of those clusters from training and re-evaluate; never remove test
images. If the test score depends heavily on those clusters, report both raw and
leakage-adjusted numbers.

## 5. Data quality issues

- Corrupt files: **{len(corrupt)}** (see `corrupt_files.csv`).
- Unlabeled images: **{sum(unlabeled_counts.values())}**; excluded but worth a manual glance
  in case they can be labelled cheaply.
- Identical/near-identical images across splits: **{cross_clusters:,} clusters**, the main
  threat to a trustworthy test score.
- No capture metadata (farm, date, camera, bird ID), so random splits measure
  within-dataset generalization only.

## 6. Modeling implications

1. **Use macro F1 and per-class recall**, not accuracy: `{largest_label}` alone is
   {100.0 * train_counts.max() / totals['train']:.1f}% of train.
2. **Class weights or a balanced sampler** are required; plain cross-entropy will ignore
   `{smallest_label}`.
3. **Deduplicate before splitting/training.** Prefer the provided split, but quarantine
   train-side members of cross-split clusters. Document the decision.
4. **Report bootstrap confidence intervals**; with {totals['test']:,} test images, rare-class
   estimates are fragile.
5. **Augment for capture conditions** (brightness, contrast, color temperature, blur) if the
   quality gaps above are class-correlated.
6. **Do not trust a single accuracy number** until Week 4 Grad-CAM review shows the model uses
   lesions/plumage, not bedding or lighting.
7. **Keep the `Unlabeled` bucket out** of training and evaluation; label it only if a
   separate manual-labelling effort is funded.
8. **Resolution is capped at 220×220**; test 224–256 px training and note the ceiling.

## 7. Sample images

![Sample grid](figures/sample_grid.png)
![Image sizes](figures/image_sizes.png)

## 8. Week 1 exit checklist

- [x] Inventory reconciles to **{len(inv):,}** images across {len(SPLITS)} splits
- [x] Class distribution and imbalance quantified
- [x] Corruption scan completed ({len(corrupt)} unreadable)
- [x] Exact and near-duplicate audit completed
- [x] Cross-split leakage quantified and policy written
- [x] Image-quality and color confounds summarized
- [x] Data card drafted
- [x] Modeling implications handed to Week 2

## 9. Artifacts

| File | Contents |
| --- | --- |
| `image_inventory.csv` | Per-image metadata, quality metrics, hashes |
| `class_distribution.csv` | Counts and percentages by split and class |
| `quality_by_class.csv` | Brightness/contrast/sharpness/color stats |
| `image_sizes.csv` | Dimension counts |
| `corrupt_files.csv` | Unreadable files |
| `duplicate_clusters.csv` | Exact/near-duplicate clusters |
| `near_duplicates.csv` | Pairwise near-duplicate matches |
| `leakage_by_split_label.csv` | Cross-split leakage by split and class |
| `summary_metrics.json` | Machine-readable headline numbers |
| `DATA_CARD.md` | Dataset documentation for release |
"""
    if not top_pairs.empty:
        text += "\n### Closest cross-split near-duplicate pairs\n\n" + md_table(
            top_pairs.assign(distance=top_pairs["distance"].astype(int))
        ) + "\n"
    out.write_text(text)


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    out = args.out
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    items = find_images(root)
    if not items:
        raise SystemExit(f"No images found under {root}")
    print(f"Scanning {len(items):,} images ...")
    records = []
    for i, (path, split, label) in enumerate(items, 1):
        records.append(analyze_image(path, split, label, root))
        if i % 500 == 0 or i == len(items):
            print(f"  {i:,}/{len(items):,}")

    inv = pd.DataFrame([asdict(r) for r in records])
    labels = sorted(inv["label"].unique())
    labels = [l for l in labels if l != UNLABELED] + ([UNLABELED] if UNLABELED in labels else [])
    inv.to_csv(out / "image_inventory.csv", index=False)

    corrupt = inv[inv["error"] != ""][["path", "split", "label", "error"]]
    corrupt.to_csv(out / "corrupt_files.csv", index=False)

    dist = wide_distribution(inv, labels)
    long_dist = inv.groupby(["split", "label"]).size().reset_index(name="count")
    long_dist["pct_of_split"] = long_dist.apply(
        lambda r: round(100.0 * r["count"] / int((inv["split"] == r["split"]).sum()), 2), axis=1)
    long_dist.to_csv(out / "class_distribution.csv", index=False)

    sizes = inv.groupby(["width", "height"]).size().reset_index(name="count")
    sizes.to_csv(out / "image_sizes.csv", index=False)

    quality = inv[inv["error"] == ""].groupby("label").agg(
        n=("path", "size"),
        brightness_median=("brightness", "median"),
        brightness_iqr=("brightness", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        contrast_median=("contrast", "median"),
        sharpness_median=("sharpness", "median"),
        sharpness_iqr=("sharpness", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        mean_r=("mean_r", "mean"),
        mean_g=("mean_g", "mean"),
        mean_b=("mean_b", "mean"),
    ).reindex(labels).reset_index()
    quality.to_csv(out / "quality_by_class.csv", index=False)

    print("Finding near-duplicates ...")
    pairs = find_near_duplicates(inv, args.dup_threshold)
    pairs.to_csv(out / "near_duplicates.csv", index=False)
    clusters = build_clusters(inv, pairs)
    clusters.to_csv(out / "duplicate_clusters.csv", index=False)
    leakage = leakage_table(inv, clusters)
    leakage.to_csv(out / "leakage_by_split_label.csv", index=False)

    print("Plotting ...")
    plot_class_distribution(inv, labels, fig_dir / "class_distribution.png")
    plot_class_proportions(inv, labels, fig_dir / "split_composition.png")
    plot_quality(inv, labels, fig_dir / "quality_by_class.png")
    plot_color_profile(quality, labels, fig_dir / "color_profile.png")
    plot_sample_grid(inv, labels, root, args.sample_per_class, args.seed,
                     fig_dir / "sample_grid.png")
    plot_leakage(leakage, fig_dir / "leakage.png")
    plot_duplicate_examples(pairs, root, fig_dir / "duplicate_examples.png")
    plot_image_sizes(inv, fig_dir / "image_sizes.png")

    totals = {split: int((inv["split"] == split).sum()) for split in SPLITS}
    supervised = [l for l in labels if l != UNLABELED]
    train_counts = dist.set_index("label")["train"].reindex(supervised)
    exact_groups = inv[inv["error"] == ""].groupby("md5").size()
    exact_groups = exact_groups[exact_groups > 1]
    leak_test = leakage[leakage["split"] == "test"]
    leak_test_images = int(leak_test["images_in_cross_split_clusters"].sum())

    metrics = {
        "dataset": "Poultry Disease Detection v9",
        "n_images": int(len(inv)),
        "splits": totals,
        "labels": labels,
        "supervised_labels": supervised,
        "n_classes": len(supervised),
        "class_counts": {
            label: {split: int(((inv["split"] == split) & (inv["label"] == label)).sum())
                    for split in SPLITS}
            for label in labels
        },
        "unlabeled_counts": {
            split: int(((inv["split"] == split) & (inv["label"] == UNLABELED)).sum())
            for split in SPLITS
        },
        "imbalance_ratio_train": round(float(train_counts.max() / max(train_counts.min(), 1)), 2),
        "largest_train_class": str(train_counts.idxmax()),
        "smallest_train_class": str(train_counts.idxmin()),
        "unique_image_sizes": int(len(sizes)),
        "corrupt_files": int(len(corrupt)),
        "exact_duplicate_groups": int(len(exact_groups)),
        "exact_duplicate_images": int(exact_groups.sum()) if len(exact_groups) else 0,
        "near_duplicate_pairs": int(len(pairs)),
        "cross_split_duplicate_pairs": int(pairs["cross_split"].sum()) if not pairs.empty else 0,
        "cross_split_clusters": int((clusters["n_splits"] > 1).sum()) if not clusters.empty else 0,
        "mixed_label_clusters": int(clusters["mixed_labels"].sum()) if not clusters.empty else 0,
        "valid_images_in_cross_split_clusters": int(
            leakage.loc[leakage["split"] == "valid", "images_in_cross_split_clusters"].sum()),
        "test_images_in_cross_split_clusters": leak_test_images,
        "test_leakage_pct": round(100.0 * leak_test_images / totals["test"], 2) if totals["test"] else 0.0,
        "dup_threshold_bits": args.dup_threshold,
        "quality_by_class": {
            row.label: {
                "n": int(row.n),
                "brightness_median": round(float(row.brightness_median), 2),
                "sharpness_median": round(float(row.sharpness_median), 2),
            }
            for row in quality.itertuples()
        },
    }
    (out / "summary_metrics.json").write_text(json.dumps(metrics, indent=2))

    write_data_card(inv, dist, labels, root, out / "DATA_CARD.md")
    write_summary(inv, dist, quality, pairs, clusters, leakage, corrupt,
                  args.dup_threshold, out / "eda_summary.md")

    print("\n=== Week 1 EDA complete ===")
    print(f"Images: {len(inv):,} | train {totals['train']:,} | valid {totals['valid']:,} | test {totals['test']:,}")
    print(f"Supervised classes: {len(supervised)} | corrupt: {len(corrupt)}")
    print(f"Near-duplicate pairs: {len(pairs):,} | cross-split clusters: "
          f"{metrics['cross_split_clusters']:,} | test leakage: {metrics['test_leakage_pct']}%")
    print(f"Artifacts: {out}/ (open {out / 'eda_summary.md'})")


if __name__ == "__main__":
    main()
