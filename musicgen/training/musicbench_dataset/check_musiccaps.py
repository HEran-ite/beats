#!/usr/bin/env python3
from datasets import load_dataset

print("Loading MusicCaps...")
dataset = load_dataset("google/MusicCaps")

print(f"Splits: {dataset.keys()}")
for split in dataset.keys():
    print(f"\n{split} split: {len(dataset[split])} samples")
    sample = dataset[split][0]
    print(f"Keys: {sample.keys()}")
    for key, value in sample.items():
        print(f"  {key}: {type(value)} = {str(value)[:150]}")
