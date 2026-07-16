import argparse
import csv
import json
import os
from pathlib import Path


def _iter_image_files(root_dir, pattern):
    if not root_dir.exists():
        return []
    if pattern is None:
        return sorted([p for p in root_dir.rglob('*') if p.is_file()])
    return sorted([p for p in root_dir.rglob(pattern) if p.is_file()])


def _find_site_roots(root_dir, ground_dir, reference_dir, maxar_dir):
    root = Path(root_dir)
    site_roots = []

    def _is_site_dir(candidate):
        return (candidate / ground_dir).exists() or (candidate / reference_dir).exists() or (candidate / maxar_dir).exists()

    if _is_site_dir(root):
        site_roots.append(root)

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if _is_site_dir(child):
            site_roots.append(child)

    # If there are no direct site directories, fall back to scanning one level deeper for
    # directories that contain the expected subfolders.
    if not site_roots:
        for candidate in sorted(root.rglob('*'), key=lambda p: str(p)):
            if candidate.is_dir() and _is_site_dir(candidate):
                site_roots.append(candidate)

    return sorted(dict.fromkeys(site_roots), key=lambda p: str(p))


def _find_matching_reference(ground_path, ref_root):
    stem = ground_path.stem
    candidates = []
    for path in ref_root.rglob('*.json'):
        if path.is_file() and (path.stem == stem or path.stem.startswith(stem) or stem.startswith(path.stem)):
            candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)), reverse=False)[0]

    # Fallback: pick the first JSON under that site's reference tree.
    json_candidates = list(ref_root.rglob('*.json'))
    return json_candidates[0] if json_candidates else None


def _find_maxar_tif(maxar_root):
    tif_candidates = []
    for ext in ('*.tif', '*.tiff'):
        tif_candidates.extend(maxar_root.rglob(ext))
    return sorted(tif_candidates)[0] if tif_candidates else None


def _find_maxar_json(maxar_root):
    json_candidates = list(maxar_root.rglob('*.json'))
    return json_candidates[0] if json_candidates else ''


def _assign_site_splits(site_names, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if not site_names:
        return {}
    site_names = sorted(site_names)
    num_sites = len(site_names)
    if num_sites == 1:
        return {site_names[0]: 'train'}

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        train_ratio = val_ratio = test_ratio = 1.0 / 3.0
    else:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    train_count = max(1, int(round(num_sites * train_ratio)))
    val_count = max(1, int(round(num_sites * val_ratio))) if num_sites > 2 else 0
    test_count = num_sites - train_count - val_count
    if test_count < 0:
        val_count = max(0, val_count + test_count)
        test_count = num_sites - train_count - val_count
    if test_count < 0:
        train_count = max(0, train_count + test_count)
        test_count = num_sites - train_count - val_count
    if test_count < 0:
        test_count = 0

    if train_count == 0 and num_sites > 0:
        train_count = 1
        if train_count + val_count > num_sites:
            val_count = max(0, num_sites - train_count)
    if val_count == 0 and num_sites > 2:
        val_count = 1
        if train_count + val_count > num_sites:
            train_count = max(0, num_sites - val_count)

    assignments = {}
    for idx, site_name in enumerate(site_names):
        if idx < train_count:
            split_name = 'train'
        elif idx < train_count + val_count:
            split_name = 'val'
        else:
            split_name = 'test'
        assignments[site_name] = split_name
    return assignments


def _write_manifest(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['ground_path', 'reference_json', 'sat_path', 'sat_json', 'label', 'site', 'lat', 'lon', 'crop_size', 'split']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {out_path}')


def build_manifest(root_dir, out_csv, ground_dir='ground', reference_dir='reference', maxar_dir='maxar', pattern='*.jpg', split=False, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'Root dir not found: {root}')

    rows = []
    site_roots = _find_site_roots(root, ground_dir, reference_dir, maxar_dir)
    if not site_roots:
        raise FileNotFoundError(
            f'Could not find any site directories under {root} that contain {ground_dir}, {reference_dir}, or {maxar_dir}'
        )

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

        maxar_tif = _find_maxar_tif(maxar_root)
        maxar_json = _find_maxar_json(maxar_root)
        if maxar_tif is None:
            continue

        for ground_path in ground_files:
            if ground_path.is_dir():
                continue

            ref_json = _find_matching_reference(ground_path, ref_root)
            if ref_json is None:
                continue

            # Try to read GPS from the reference JSON so the dataset can crop from the Maxar TIFF later.
            lat = lon = None
            try:
                with open(ref_json, encoding='utf-8') as f:
                    ref_payload = json.load(f)
                extrinsics = ref_payload.get('extrinsics', {})
                lat = extrinsics.get('lat')
                lon = extrinsics.get('lon')
            except Exception:
                pass

            rel_ground = os.path.relpath(ground_path, root)
            rel_ref = os.path.relpath(ref_json, root)
            rel_maxar = os.path.relpath(maxar_tif, root)
            rel_maxar_json = os.path.relpath(maxar_json, root) if maxar_json else ''

            rows.append({
                'ground_path': rel_ground,
                'reference_json': rel_ref,
                'sat_path': rel_maxar,
                'sat_json': rel_maxar_json,
                'label': ground_path.stem,
                'site': site_name,
                'lat': lat,
                'lon': lon,
                'crop_size': 384,
                'split': '',
            })

    if split:
        site_to_split = _assign_site_splits(sorted({row['site'] for row in rows}), train_ratio, val_ratio, test_ratio)
        split_rows = {'train': [], 'val': [], 'test': []}
        for row in rows:
            split_name = site_to_split.get(row['site'], 'train')
            row_copy = dict(row)
            row_copy['split'] = split_name
            split_rows[split_name].append(row_copy)
        out_path = Path(out_csv)
        base = out_path.with_suffix('')
        for split_name in ('train', 'val', 'test'):
            split_path = out_path.with_name(f'{base.name}_{split_name}{out_path.suffix}')
            _write_manifest(split_rows[split_name], split_path)
    else:
        out_path = Path(out_csv)
        _write_manifest(rows, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ground-dir', default='ground')
    parser.add_argument('--reference-dir', default='reference')
    parser.add_argument('--maxar-dir', default='maxar')
    parser.add_argument('--pattern', default='*.jpg')
    parser.add_argument('--split', action='store_true', help='Write separate train/val/test manifests by site.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    args = parser.parse_args()
    build_manifest(args.root, args.out, args.ground_dir, args.reference_dir, args.maxar_dir, args.pattern, args.split, args.train_ratio, args.val_ratio, args.test_ratio)
