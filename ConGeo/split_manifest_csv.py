import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict


def split_manifest(input_csv: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    if not (0.0 < train_ratio <= 1.0 and 0.0 < val_ratio <= 1.0 and 0.0 < test_ratio <= 1.0):
        raise ValueError('Ratios must be positive and between 0 and 1')
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    input_path = Path(input_csv)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(input_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f'No rows found in {input_path}')

    import random
    random.seed(seed)
    random.shuffle(rows)

    n = len(rows)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    fieldnames = list(rows[0].keys()) if rows else ['ground_path','reference_json','sat_path','label']

    def write_split(split_name: str, split_rows: List[Dict[str, str]]):
        out_path = output_dir_path / f'manifest_{split_name}{input_path.suffix}'
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f'Wrote {len(split_rows)} rows to {out_path}')

    write_split('train', train_rows)
    write_split('val', val_rows)
    write_split('test', test_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split an existing manifest.csv into train/val/test CSVs.')
    parser.add_argument('--input', required=True, help='Path to the input manifest CSV file.')
    parser.add_argument('--out-dir', required=True, help='Directory where train/val/test CSVs will be written.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split_manifest(args.input, args.out_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
