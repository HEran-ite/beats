#!/usr/bin/env python3
"""
Prepare looperman_audio dataset for MusicGen training
Extracts metadata from filenames and creates data.jsonl files
"""

import os
import json
import re
import random
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf

# Paths
INPUT_DIR = Path("/home/ubuntu/beats/looperman_audio")
OUTPUT_DIR = Path("/home/ubuntu/beats/musicgen/training/audiocraft/egs/looperman_audio")

# Split ratios
TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1


def parse_filename(filename):
    """
    Parse filename to extract metadata
    Examples:
    - 4992_Azul_2_Prod_ScOnTheTrack_153bpm_G#.mp3
    - 4996_To_The_Riches_-_Gunna_Guitar_Loop_148bpm_G#m.mp3
    - 4997_Sweat_89bpm.mp3
    """
    stem = Path(filename).stem
    
    # Extract BPM (look for pattern like "153bpm" or "89bpm")
    bpm_match = re.search(r'(\d+)bpm', stem, re.IGNORECASE)
    bpm = float(bpm_match.group(1)) if bpm_match else None
    
    # Extract key (look for patterns like "G#", "G#m", "Cm", "D#m", etc.)
    # Keys can be: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    # With optional 'm' for minor
    key_pattern = r'([A-G](#|b)?m?)(?=\s|$|\.|_)'
    key_match = re.search(key_pattern, stem, re.IGNORECASE)
    key = key_match.group(1) if key_match else None
    
    # Normalize key format (convert to standard format)
    if key:
        key = key.upper()
        # Convert 'b' to '#' notation (e.g., Bb -> A#)
        key_replacements = {
            'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'
        }
        for old, new in key_replacements.items():
            if key.startswith(old):
                key = key.replace(old, new)
    
    # Extract title/description (everything before bpm/key)
    title_parts = stem.split('_')
    # Remove ID at the start if it's numeric
    if title_parts and title_parts[0].isdigit():
        title_parts = title_parts[1:]
    
    # Remove bpm and key from title
    title = ' '.join(title_parts)
    title = re.sub(r'\d+bpm', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[A-G](#|b)?m?', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[_-]+', ' ', title).strip()
    
    return {
        'bpm': bpm,
        'key': key,
        'title': title
    }


def get_audio_info(audio_path):
    """Get audio file information"""
    try:
        info = sf.info(audio_path)
        duration = info.duration
        sample_rate = info.samplerate
        return {
            'duration': duration,
            'sample_rate': sample_rate
        }
    except Exception as e:
        print(f"  ⚠️  Error reading {audio_path}: {e}")
        return None


def create_description(metadata):
    """Create description from metadata"""
    parts = []
    
    if metadata.get('title'):
        parts.append(metadata['title'])
    
    if metadata.get('bpm'):
        parts.append(f"{int(metadata['bpm'])} BPM")
    
    if metadata.get('key'):
        key_str = metadata['key']
        # Add 'minor' if it ends with 'm'
        if key_str.endswith('m'):
            key_str = key_str[:-1] + " minor"
        else:
            key_str = key_str + " major"
        parts.append(f"in {key_str}")
    
    if parts:
        return ", ".join(parts)
    return "instrumental music"


def prepare_dataset():
    """Prepare looperman_audio dataset"""
    
    print("="*80)
    print("  PREPARING LOOPERMAN AUDIO DATASET")
    print("="*80)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Get all MP3 files
    audio_files = sorted(INPUT_DIR.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files\n")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # Process all files
    all_entries = []
    
    print("Processing audio files...")
    for audio_file in tqdm(audio_files, desc="Processing"):
        # Parse filename
        metadata = parse_filename(audio_file.name)
        
        # Get audio info
        audio_info = get_audio_info(audio_file)
        if audio_info is None:
            continue
        
        # Create description
        description = create_description(metadata)
        
        # Create entry
        entry = {
            'path': str(audio_file.absolute()),
            'duration': audio_info['duration'],
            'sample_rate': audio_info['sample_rate'],
            'amplitude': None,
            'weight': None,
            'info_path': None,
            'description': description,
            'meta': {
                'bpm': metadata.get('bpm'),
                'key': metadata.get('key'),
                'title': metadata.get('title')
            }
        }
        
        all_entries.append(entry)
    
    print(f"\n✅ Processed {len(all_entries)} files")
    
    # Shuffle entries
    random.seed(42)
    random.shuffle(all_entries)
    
    # Calculate splits
    n_total = len(all_entries)
    n_train = int(n_total * TRAIN_RATIO)
    n_eval = int(n_total * EVAL_RATIO)
    
    train_entries = all_entries[:n_train]
    eval_entries = all_entries[n_train:n_train + n_eval]
    test_entries = all_entries[n_train + n_eval:]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_entries)} ({len(train_entries)/n_total*100:.1f}%)")
    print(f"  Eval: {len(eval_entries)} ({len(eval_entries)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_entries)} ({len(test_entries)/n_total*100:.1f}%)")
    
    # Create output directories and write splits
    splits = {
        'train': train_entries,
        'valid': eval_entries,
        'evaluate': test_entries,
        'generate': test_entries[:10]  # First 10 from test for generation
    }
    
    for split_name, entries in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = split_dir / 'data.jsonl'
        with open(jsonl_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"  ✅ Created {jsonl_path} ({len(entries)} entries)")
    
    print(f"\n{'='*80}")
    print("✅ Dataset preparation complete!")
    print(f"{'='*80}")
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"Next step: Create dataset config and start training\n")


if __name__ == "__main__":
    prepare_dataset()


