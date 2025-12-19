#!/usr/bin/env python3
"""Inspect MusicBench dataset structure"""

from datasets import load_dataset

print("Loading MusicBench dataset...")
dataset = load_dataset("amaai-lab/MusicBench", split="train")

print(f"\nDataset size: {len(dataset)}")
print(f"\nFirst sample:")
sample = dataset[0]
print(f"Type: {type(sample)}")
print(f"Keys: {sample.keys() if hasattr(sample, 'keys') else 'N/A'}")
print(f"\nSample content:")
for key, value in sample.items():
    print(f"  {key}: {type(value)} = {str(value)[:200]}")
