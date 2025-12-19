#!/usr/bin/env python3
"""Split the full dataset manifest into train/valid/evaluate/generate splits"""

import argparse
import json
import random
from pathlib import Path


def split_manifest(input_manifest: Path, output_dir: Path, train_ratio: float = 0.8, valid_ratio: float = 0.1):
    """Split manifest into train/valid/evaluate/generate"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all entries
    entries = []
    with open(input_manifest, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    # Shuffle
    random.seed(42)
    random.shuffle(entries)
    
    # Calculate splits
    n_total = len(entries)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_entries = entries[:n_train]
    valid_entries = entries[n_train:n_train + n_valid]
    evaluate_entries = entries[n_train + n_valid:]
    generate_entries = entries[n_train + n_valid:n_train + n_valid + 10]  # First 10 from eval
    
    print(f"Total: {n_total}")
    print(f"Train: {len(train_entries)}")
    print(f"Valid: {len(valid_entries)}")
    print(f"Evaluate: {len(evaluate_entries)}")
    print(f"Generate: {len(generate_entries)}")
    
    # Write splits
    for split_name, split_entries in [
        ('train', train_entries),
        ('valid', valid_entries),
        ('evaluate', evaluate_entries),
        ('generate', generate_entries)
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_file = split_dir / 'data.jsonl'
        
        with open(split_file, 'w') as f:
            for entry in split_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Created {split_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_manifest', type=Path, help='Input JSONL manifest file')
    parser.add_argument('output_dir', type=Path, help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    split_manifest(args.input_manifest, args.output_dir, args.train_ratio, args.valid_ratio)



