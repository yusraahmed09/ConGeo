import argparse
import csv
import json
import os
from pathlib import Path


def build_manifest(root_dir, out_csv, ground_dir='ground', reference_dir='reference', maxar_dir='maxar', pattern='*.jpg'):
    root = Path(root_dir)
    ground_root = root / ground_dir
    ref_root = root / reference_dir
    maxar_root = root / maxar_dir

    rows = []
    if not ground_root.exists():
        raise FileNotFoundError(f'Ground dir not found: {ground_root}')
    if not ref_root.exists():
        raise FileNotFoundError(f'Reference dir not found: {ref_root}')
    if not maxar_root.exists():
        raise FileNotFoundError(f'Maxar dir not found: {maxar_root}')

    # Pair each ground image with the first Maxar .tif/.json pair found in the same site folder.
    # If you later want a GPS-based mapping, replace this logic with your own pairing rule.
    for ground_path in sorted(ground_root.rglob(pattern)):
        if ground_path.is_dir():
            continue

        site_name = None
        for part in ground_path.parts:
            if part.startswith('site') or part.startswith('Site'):
                site_name = part
                break
        if site_name is None:
            site_name = ground_path.parent.name

        site_ref_root = ref_root / site_name
        site_maxar_root = maxar_root / site_name

        ref_candidates = []
        if site_ref_root.exists():
            ref_candidates = list(site_ref_root.rglob(f'{ground_path.stem}*.json'))
        if not ref_candidates:
            ref_candidates = list(ref_root.rglob(f'{ground_path.stem}*.json'))
        if not ref_candidates:
            continue
        ref_json = ref_candidates[0]

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

        maxar_tif_candidates = []
        maxar_json_candidates = []
        if site_maxar_root.exists():
            maxar_tif_candidates = list(site_maxar_root.rglob('*.tif')) + list(site_maxar_root.rglob('*.tiff'))
            maxar_json_candidates = list(site_maxar_root.rglob('*.json'))
        if not maxar_tif_candidates:
            maxar_tif_candidates = list(maxar_root.rglob('*.tif')) + list(maxar_root.rglob('*.tiff'))
            maxar_json_candidates = list(maxar_root.rglob('*.json'))
        if not maxar_tif_candidates:
            continue
        maxar_tif = maxar_tif_candidates[0]
        maxar_json = maxar_json_candidates[0] if maxar_json_candidates else ''

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
        })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ground_path', 'reference_json', 'sat_path', 'sat_json', 'label', 'site', 'lat', 'lon', 'crop_size'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ground-dir', default='ground')
    parser.add_argument('--reference-dir', default='reference')
    parser.add_argument('--maxar-dir', default='maxar')
    parser.add_argument('--pattern', default='*.jpg')
    args = parser.parse_args()
    build_manifest(args.root, args.out, args.ground_dir, args.reference_dir, args.maxar_dir, args.pattern)
