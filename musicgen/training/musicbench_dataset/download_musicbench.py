#!/usr/bin/env python3
"""
Download and prepare MusicBench dataset for MusicGen training

Dataset: amaai-lab/MusicBench
- 52,768 training samples
- Multiple genres/styles
- Complete metadata: BPM, Key, Chords, Beats
- Text prompts for each track
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Paths
OUTPUT_DIR = "/home/ubuntu/beats/musicgen/training/audiocraft/egs/musicbench"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluate")

os.makedirs(os.path.join(TRAIN_DIR, "audio_files"), exist_ok=True)
os.makedirs(os.path.join(EVAL_DIR, "audio_files"), exist_ok=True)

def download_and_prepare():
    """
    Download MusicBench and convert to MusicGen format
    """
    print("📥 Loading MusicBench dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("amaai-lab/MusicBench", split="train")
    test_dataset = load_dataset("amaai-lab/MusicBench", split="test")

    print(f"✅ Loaded {len(dataset)} training samples")
    print(f"✅ Loaded {len(test_dataset)} test samples")

    # Process training data
    print("\n🔄 Processing training data...")
    train_data = []

    for idx, sample in enumerate(tqdm(dataset)):
        try:
            # Extract metadata
            bpm = sample.get('bpm', 120)  # Default to 120 if missing
            key = sample.get('key', 'C major')  # Default key
            caption = sample.get('caption', '')
            location = sample.get('location', '')

            # Save audio file
            audio_filename = f"track_{idx:06d}.wav"
            audio_path = os.path.join(TRAIN_DIR, "audio_files", audio_filename)

            # Get audio array from dataset
            if 'audio' in sample and sample['audio'] is not None:
                audio_data = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']

                # Save as WAV
                sf.write(audio_path, audio_data, sample_rate)

                # Create metadata entry
                entry = {
                    "path": f"audio_files/{audio_filename}",
                    "bpm": float(bpm) if bpm else 120.0,
                    "key": str(key) if key else "C major",
                    "genre": "Mixed",  # MusicBench has multiple genres
                    "prompt": caption if caption else f"Music track at {bpm} BPM in {key}",
                    "duration": len(audio_data) / sample_rate,
                    "location": location
                }

                train_data.append(entry)

        except Exception as e:
            print(f"⚠️  Error processing sample {idx}: {e}")
            continue

    # Save training metadata
    train_jsonl = os.path.join(TRAIN_DIR, "data.jsonl")
    with open(train_jsonl, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Saved {len(train_data)} training samples to {train_jsonl}")

    # Process test data (use first 100 samples for evaluation)
    print("\n🔄 Processing evaluation data...")
    eval_data = []

    for idx, sample in enumerate(tqdm(test_dataset[:100])):
        try:
            bpm = sample.get('bpm', 120)
            key = sample.get('key', 'C major')
            caption = sample.get('caption', '')
            location = sample.get('location', '')

            audio_filename = f"track_{idx:06d}.wav"
            audio_path = os.path.join(EVAL_DIR, "audio_files", audio_filename)

            if 'audio' in sample and sample['audio'] is not None:
                audio_data = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']

                sf.write(audio_path, audio_data, sample_rate)

                entry = {
                    "path": f"audio_files/{audio_filename}",
                    "bpm": float(bpm) if bpm else 120.0,
                    "key": str(key) if key else "C major",
                    "genre": "Mixed",
                    "prompt": caption if caption else f"Music track at {bpm} BPM in {key}",
                    "duration": len(audio_data) / sample_rate,
                    "location": location
                }

                eval_data.append(entry)

        except Exception as e:
            print(f"⚠️  Error processing eval sample {idx}: {e}")
            continue

    # Save evaluation metadata
    eval_jsonl = os.path.join(EVAL_DIR, "data.jsonl")
    with open(eval_jsonl, 'w') as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Saved {len(eval_data)} evaluation samples to {eval_jsonl}")

    # Print summary
    print("\n" + "="*80)
    print("📊 DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    print(f"\nTraining data: {TRAIN_DIR}")
    print(f"Evaluation data: {EVAL_DIR}")
    print("\n✅ Ready for MusicGen training!")

if __name__ == "__main__":
    download_and_prepare()
