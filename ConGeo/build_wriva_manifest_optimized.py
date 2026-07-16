import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover
    rasterio = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SUPPORTED_MAXAR_SUFFIXES = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def _is_site_container(candidate: Path, ground_dir: str, reference_dir: str, maxar_dir: str) -> bool:
    return (candidate / ground_dir).exists() or (candidate / reference_dir).exists() or (candidate / maxar_dir).exists()


def discover_site_roots(root_dir: Path, ground_dir: str, reference_dir: str, maxar_dir: str) -> List[Path]:
    root_dir = root_dir.resolve()
    discovered: List[Path] = []
    seen = set()

    def add_if_needed(candidate: Path) -> None:
        if not candidate.exists() or not candidate.is_dir():
            return
        key = str(candidate)
        if key in seen:
            return
        if _is_site_container(candidate, ground_dir, reference_dir, maxar_dir):
            discovered.append(candidate)
            seen.add(key)

    add_if_needed(root_dir)

    for child in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            add_if_needed(child)

    if discovered:
        return sorted(dict.fromkeys(discovered), key=lambda p: str(p))

    for current_root, dirnames, _ in os.walk(root_dir):
        candidate = Path(current_root)
        add_if_needed(candidate)
        dirnames.sort()

    return sorted(dict.fromkeys(discovered), key=lambda p: str(p))


def _iter_image_files(root_dir: Path, pattern: str) -> List[Path]:
    if not root_dir.exists():
        return []
    if not pattern:
        return sorted([p for p in root_dir.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])
    return sorted([p for p in root_dir.rglob(pattern) if p.is_file()])


def _read_small_image(path: Path):
    try:
        if path.suffix.lower() in {'.tif', '.tiff'}:
            if rasterio is not None:
                with rasterio.open(path) as src:
                    width = min(src.width, 256)
                    height = min(src.height, 256)
                    data = src.read(window=Window(0, 0, width, height))
                    if data is None:
                        return None
                    if data.ndim == 2:
                        return data
                    if data.shape[0] in (1, 3, 4):
                        data = np.moveaxis(data, 0, -1)
                    return data
            if tifffile is not None:
                img = tifffile.imread(path)
                if img is None:
                    return None
                if img.ndim == 2:
                    return img
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = np.moveaxis(img, 0, -1)
                return img
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image
            with Image.open(path) as im:
                return np.array(im.convert('RGB'))
        except Exception:
            return None
    except Exception:
        return None


def is_valid_satellite_asset(path: Path, bright_threshold: int = 240, blank_ratio: float = 0.97, dark_threshold: int = 15) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in SUPPORTED_MAXAR_SUFFIXES:
        return False
    try:
        img = _read_small_image(path)
        if img is None:
            return False
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
        else:
            gray = img
        if gray.size == 0:
            return False
        bright_ratio = (gray >= bright_threshold).mean()
        dark_ratio = (gray <= dark_threshold).mean()
        variance = float(gray.var()) if hasattr(gray, 'var') else 0.0
        if bright_ratio > blank_ratio or dark_ratio > blank_ratio or variance < 1e-3:
            return False
        return True
    except Exception:
        return False


def extract_gps_from_reference(ref_json: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if not ref_json or not ref_json.exists():
        return None, None
    try:
        with open(ref_json, encoding='utf-8') as f:
            payload = json.load(f)
        extrinsics = payload.get('extrinsics', {})
        lat = extrinsics.get('lat')
        lon = extrinsics.get('lon')
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)
    except Exception:
        return None, None


def find_matching_reference_json(ground_path: Path, ref_root: Path) -> Optional[Path]:
    stem = ground_path.stem
    candidates: List[Path] = []
    for path in ref_root.rglob('*.json'):
        if not path.is_file():
            continue
        candidate_stem = path.stem
        if candidate_stem == stem or candidate_stem.startswith(stem) or stem.startswith(candidate_stem):
            candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]
    json_candidates = [p for p in ref_root.rglob('*.json') if p.is_file()]
    return json_candidates[0] if json_candidates else None


def find_matching_sidecar_json(sat_path: Path, maxar_root: Path) -> Optional[Path]:
    for suffix in ('.json', '.JSON'):
        candidate = sat_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    for path in maxar_root.rglob('*.json'):
        if path.is_file() and path.stem == sat_path.stem:
            return path
    return None


def gather_valid_satellite_assets(maxar_root: Path, max_variants: Optional[int] = None, seed: Optional[int] = None) -> List[Path]:
    if not maxar_root.exists():
        return []
    candidates = []
    for path in sorted(maxar_root.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_MAXAR_SUFFIXES:
            continue
        if is_valid_satellite_asset(path):
            candidates.append(path)
    if seed is not None:
        random.seed(seed)
    if max_variants is not None and max_variants > 0 and len(candidates) > max_variants:
        candidates = random.sample(candidates, max_variants)
    else:
        candidates = sorted(candidates, key=lambda p: str(p))
    return candidates


def assign_site_splits(site_names: List[str], train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, str]:
    if not site_names:
        return {}
    ordered = sorted(site_names)
    n = len(ordered)
    if n == 1:
        return {ordered[0]: 'train'}

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        train_ratio = val_ratio = test_ratio = 1.0 / 3.0
    else:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    train_count = max(1, int(round(n * train_ratio)))
    if n > 2:
        val_count = max(1, int(round(n * val_ratio)))
    else:
        val_count = 0
    test_count = n - train_count - val_count
    if test_count < 0:
        val_count = max(0, val_count + test_count)
        test_count = n - train_count - val_count
    if test_count < 0:
        train_count = max(0, train_count + test_count)
        test_count = n - train_count - val_count
    if train_count == 0 and n > 0:
        train_count = 1
        if train_count + val_count > n:
            val_count = max(0, n - train_count)
    if val_count == 0 and n > 2:
        val_count = 1
        if train_count + val_count > n:
            train_count = max(0, n - val_count)

    assignments: Dict[str, str] = {}
    for idx, site_name in enumerate(ordered):
        if idx < train_count:
            split_name = 'train'
        elif idx < train_count + val_count:
            split_name = 'val'
        else:
            split_name = 'test'
        assignments[site_name] = split_name
    return assignments


def write_manifest(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'ground_path',
        'reference_json',
        'sat_path',
        'sat_json',
        'label',
        'site',
        'lat',
        'lon',
        'crop_size',
        'weather_variant',
        'split',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {out_path}')


def build_manifest(
    root_dir: str,
    out_csv: str,
    ground_dir: str = 'ground',
    reference_dir: str = 'reference',
    maxar_dir: str = 'maxar',
    pattern: str = '*.jpg',
    split: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    crop_size: int = 384,
    max_ground_per_site: Optional[int] = None,
    max_variants_per_site: Optional[int] = None,
    max_pairs_per_site: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f'Root dir not found: {root}')

    site_roots = discover_site_roots(root, ground_dir, reference_dir, maxar_dir)
    if not site_roots:
        raise FileNotFoundError(
            f'Could not find any site directories under {root} that contain {ground_dir}, {reference_dir}, or {maxar_dir}'
        )

    rows: List[Dict[str, object]] = []
    site_to_split: Optional[Dict[str, str]] = None

    for site_root in site_roots:
        ground_root = site_root / ground_dir
        ref_root = site_root / reference_dir
        maxar_root = site_root / maxar_dir
        if not ground_root.exists() or not ref_root.exists() or not maxar_root.exists():
            continue

        site_name = site_root.name if site_root != root else root.name
        ground_files = _iter_image_files(ground_root, pattern)
        if not ground_files:
            continue

        if max_ground_per_site is not None and max_ground_per_site > 0 and len(ground_files) > max_ground_per_site:
            if seed is not None:
                random.seed(seed + hash(site_name) % 100000)
            ground_files = random.sample(ground_files, max_ground_per_site)
        else:
            ground_files = sorted(ground_files, key=lambda p: str(p))

        valid_satellites = gather_valid_satellite_assets(maxar_root, max_variants=max_variants_per_site, seed=seed)
        if not valid_satellites:
            continue

        pair_budget = max_pairs_per_site if max_pairs_per_site is not None else None
        pairs_emitted = 0
        for ground_path in ground_files:
            if ground_path.is_dir():
                continue
            ref_json = find_matching_reference_json(ground_path, ref_root)
            lat, lon = extract_gps_from_reference(ref_json)
            for sat_path in valid_satellites:
                if pair_budget is not None and pairs_emitted >= pair_budget:
                    break
                rel_ground = os.path.relpath(ground_path, root)
                rel_ref = os.path.relpath(ref_json, root) if ref_json else ''
                rel_maxar = os.path.relpath(sat_path, root)
                sat_json = find_matching_sidecar_json(sat_path, maxar_root)
                rel_maxar_json = os.path.relpath(sat_json, root) if sat_json else ''
                weather_variant = sat_path.parent.name if sat_path.parent != maxar_root else sat_path.stem
                rows.append({
                    'ground_path': rel_ground,
                    'reference_json': rel_ref,
                    'sat_path': rel_maxar,
                    'sat_json': rel_maxar_json,
                    'label': ground_path.stem,
                    'site': site_name,
                    'lat': lat,
                    'lon': lon,
                    'crop_size': crop_size,
                    'weather_variant': weather_variant,
                    'split': '',
                })
                pairs_emitted += 1
            if pair_budget is not None and pairs_emitted >= pair_budget:
                break

    if split:
        site_to_split = assign_site_splits(sorted({str(row['site']) for row in rows}), train_ratio, val_ratio, test_ratio)
        split_rows = {'train': [], 'val': [], 'test': []}
        for row in rows:
            split_name = site_to_split.get(str(row['site']), 'train')
            row_copy = dict(row)
            row_copy['split'] = split_name
            split_rows[split_name].append(row_copy)

        out_path = Path(out_csv)
        base = out_path.with_suffix('')
        for split_name in ('train', 'val', 'test'):
            split_path = out_path.with_name(f'{base.name}_{split_name}{out_path.suffix}')
            write_manifest(split_rows[split_name], split_path)
    else:
        write_manifest(rows, Path(out_csv))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an optimized WRIVA-style manifest with per-ground-image pairing to all valid weather variants.')
    parser.add_argument('--root', required=True, help='Root directory containing site folders with ground/reference/maxar subfolders.')
    parser.add_argument('--out', required=True, help='Output manifest path (.csv).')
    parser.add_argument('--ground-dir', default='ground')
    parser.add_argument('--reference-dir', default='reference')
    parser.add_argument('--maxar-dir', default='maxar')
    parser.add_argument('--pattern', default='*.jpg', help='Glob pattern for ground images, e.g. *.jpg or *.png')
    parser.add_argument('--split', action='store_true', help='Write train/val/test manifests per site.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--max-ground-per-site', type=int, default=500, help='Maximum number of ground images to sample from each site.')
    parser.add_argument('--max-variants-per-site', type=int, default=8, help='Maximum number of valid Maxar variants to use per site.')
    parser.add_argument('--max-pairs-per-site', type=int, default=4000, help='Maximum number of ground-image/variant pairs to emit per site.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic per-site sampling.')
    args = parser.parse_args()
    build_manifest(
        args.root,
        args.out,
        ground_dir=args.ground_dir,
        reference_dir=args.reference_dir,
        maxar_dir=args.maxar_dir,
        pattern=args.pattern,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        crop_size=args.crop_size,
        max_ground_per_site=args.max_ground_per_site,
        max_variants_per_site=args.max_variants_per_site,
        max_pairs_per_site=args.max_pairs_per_site,
        seed=args.seed,
    )
