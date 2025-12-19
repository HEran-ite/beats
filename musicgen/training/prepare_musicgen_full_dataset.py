#!/usr/bin/env python3
"""
Prepare full MusicGen dataset with proper text descriptions
Uses all 1,465 beats with 80/10/10 split (same as ACE-Step)
"""

import os
import json
import random
import csv
from pathlib import Path
from typing import Dict, List
import librosa

# Set random seed for reproducible splits
random.seed(42)

# Paths
BEATS_DIR = "/home/ubuntu/amanuel/beats/downloaded_beats 2"
MUSIC_DATASET_DIR = "/home/ubuntu/amanuel/beats/music_dataset/datasets"
METADATA_CSV = "/home/ubuntu/amanuel/beats/music_dataset/metadata/beats_data.csv"
OUTPUT_DIR = "/home/ubuntu/amanuel/beats/musicgen/training/audiocraft/egs/beats_full_complete"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_csv_metadata() -> Dict[str, Dict]:
    """Load all metadata from CSV into a dictionary keyed by UUID"""
    metadata_dict = {}

    if not os.path.exists(METADATA_CSV):
        print(f"Warning: CSV metadata not found at {METADATA_CSV}")
        return metadata_dict

    with open(METADATA_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row.get('id', '')
            if uuid:
                metadata_dict[uuid] = {
                    'name': row.get('name', ''),
                    'bpm': float(row['bpm']) if row.get('bpm') else None,
                    'key': row.get('key', ''),
                    'style': row.get('style', ''),
                    'genre': row.get('style', ''),  # Alias
                    'artist': row.get('artist', ''),
                    'length_seconds': float(row['length_seconds']) if row.get('length_seconds') else None
                }

    print(f"✅ Loaded metadata for {len(metadata_dict)} beats from CSV")
    return metadata_dict


def load_metadata(audio_file: str, csv_metadata: Dict[str, Dict]) -> Dict:
    """Load metadata for a specific audio file"""
    # Extract UUID from filename
    basename = os.path.basename(audio_file)
    uuid = basename.replace('.mp3', '')

    # Check CSV metadata first
    if uuid in csv_metadata:
        return csv_metadata[uuid]

    # Fallback: Try JSON file in same directory
    json_file = audio_file.replace('.mp3', '.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    # Fallback: Try music_dataset directory
    music_json = os.path.join(MUSIC_DATASET_DIR, basename.replace('.mp3', '.json'))
    if os.path.exists(music_json):
        with open(music_json, 'r') as f:
            return json.load(f)

    return {}


def create_description(metadata: Dict, filename: str) -> str:
    """Create text description from metadata"""
    parts = []

    # Title/Name
    title = metadata.get('title') or metadata.get('name', '')
    if title:
        # Clean up encoding issues
        title = title.encode('ascii', 'ignore').decode('ascii').strip()
        if title:
            parts.append(title)

    # Genre/Style
    genre = metadata.get('genre') or metadata.get('style', '')
    if genre:
        parts.append(f"{genre} style")

    # BPM
    bpm = metadata.get('bpm')
    if bpm:
        parts.append(f"{int(bpm)} BPM")

    # Key
    key = metadata.get('key', '')
    if key:
        parts.append(f"in {key}")

    # Mood/Vibe
    mood = metadata.get('mood', '')
    vibe = metadata.get('vibe', '')
    if mood and mood != 'neutral':
        parts.append(f"{mood} mood")
    if vibe and vibe != 'modern':
        parts.append(f"{vibe} vibe")

    # Energy
    energy = metadata.get('energy', '')
    if energy and energy != 'medium':
        parts.append(f"{energy} energy")

    # Combine all parts
    if parts:
        description = ", ".join(parts) + "."
    else:
        # Fallback if no metadata
        description = f"Instrumental beat {filename}"

    return description


def get_audio_info(audio_path: str) -> Dict:
    """Get audio file information"""
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Just load 1 second for speed
        duration = librosa.get_duration(path=audio_path)

        return {
            'duration': duration,
            'sample_rate': sr
        }
    except Exception as e:
        print(f"Warning: Could not load {audio_path}: {e}")
        return {
            'duration': 30.0,  # Default
            'sample_rate': 32000
        }


def main():
    print("="*80)
    print("PREPARING FULL MUSICGEN DATASET")
    print("="*80)

    # Load CSV metadata first
    print(f"\n📊 Loading metadata from CSV...")
    csv_metadata = load_csv_metadata()

    # Find all MP3 files
    print(f"\n📁 Scanning: {BEATS_DIR}")
    audio_files = list(Path(BEATS_DIR).glob("*.mp3"))
    print(f"✅ Found {len(audio_files)} audio files")

    # Shuffle for random split
    random.shuffle(audio_files)

    # Calculate split sizes
    total = len(audio_files)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    train_files = audio_files[:train_size]
    val_files = audio_files[train_size:train_size + val_size]
    test_files = audio_files[train_size + val_size:]

    print(f"\n📊 Split:")
    print(f"  Train: {len(train_files)} samples ({len(train_files)/total*100:.1f}%)")
    print(f"  Valid: {len(val_files)} samples ({len(val_files)/total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} samples ({len(test_files)/total*100:.1f}%)")

    # Create output directories
    splits = {
        'train': train_files,
        'valid': val_files,
        'evaluate': test_files,
        'generate': test_files[:10]  # Use first 10 test samples for generation
    }

    for split_name, files in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n🔄 Processing {split_name} split ({len(files)} files)...")

        data_entries = []

        for i, audio_file in enumerate(files, 1):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(files)}")

            # Load metadata
            metadata = load_metadata(str(audio_file), csv_metadata)

            # Create description
            description = create_description(metadata, audio_file.stem)

            # Get audio info
            audio_info = get_audio_info(str(audio_file))

            # Create data entry
            entry = {
                'path': str(audio_file.absolute()),
                'duration': audio_info['duration'],
                'sample_rate': audio_info['sample_rate'],
                'amplitude': None,
                'weight': None,
                'info_path': None
            }

            # Add description for MusicGen text conditioning
            if description:
                entry['description'] = description

            # Add structured attributes (optional, for future use)
            if metadata:
                entry['meta'] = {
                    'genre': metadata.get('genre', ''),
                    'bpm': metadata.get('bpm', 0),
                    'key': metadata.get('key', ''),
                    'mood': metadata.get('mood', ''),
                    'energy': metadata.get('energy', '')
                }

            data_entries.append(entry)

        # Write data.jsonl
        jsonl_path = os.path.join(split_dir, 'data.jsonl')
        with open(jsonl_path, 'w') as f:
            for entry in data_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"  ✅ Saved {len(data_entries)} entries to {jsonl_path}")

        # Print sample descriptions
        if data_entries:
            print(f"\n  📝 Sample descriptions from {split_name}:")
            for entry in data_entries[:3]:
                desc = entry.get('description', 'No description')
                filename = Path(entry['path']).name
                print(f"    • {filename}: {desc}")

    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"🎵 Total samples: {total}")
    print(f"   Train: {len(train_files)}")
    print(f"   Valid: {len(val_files)}")
    print(f"   Test:  {len(test_files)}")
    print(f"   Generate: {len(splits['generate'])}")
    print("\n✨ Dataset is ready for MusicGen training!")


if __name__ == "__main__":
    main()
