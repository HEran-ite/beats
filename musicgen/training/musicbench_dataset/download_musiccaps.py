#!/usr/bin/env python3
"""
Download and prepare Google MusicCaps dataset for MusicGen training

Dataset: google/MusicCaps
- 5,521 music clips
- High-quality captions
- Actual audio included
- Multiple genres
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf
import librosa
import essentia.standard as es

# Paths
OUTPUT_DIR = "/home/ubuntu/beats/musicgen/training/audiocraft/egs/musiccaps"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluate")

os.makedirs(os.path.join(TRAIN_DIR, "audio_files"), exist_ok=True)
os.makedirs(os.path.join(EVAL_DIR, "audio_files"), exist_ok=True)

def detect_key_bpm(audio, sr):
    """Detect key and BPM using Essentia"""
    try:
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)

        # Key detection
        key_extractor = es.KeyExtractor()
        key, scale, confidence = key_extractor(audio)
        key_str = f"{key} {scale}"

        # BPM detection
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)

        return key_str, float(bpm)
    except Exception as e:
        print(f"  Warning: Could not detect key/BPM: {e}")
        return "C major", 120.0

def download_and_prepare():
    """
    Download MusicCaps and convert to MusicGen format
    """
    print("📥 Loading Google MusicCaps dataset from Hugging Face...")

    # Load dataset - MusicCaps has only one split
    try:
        dataset = load_dataset("google/MusicCaps")

        # Combine all splits
        all_samples = []
        for split_name in dataset.keys():
            all_samples.extend(list(dataset[split_name]))

        print(f"✅ Loaded {len(all_samples)} total samples")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Split into train (90%) and eval (10%)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]

    print(f"📊 Split: {len(train_samples)} training, {len(eval_samples)} evaluation")

    # Process training data
    print("\n🔄 Processing training data...")
    train_data = []

    for idx, sample in enumerate(tqdm(train_samples)):
        try:
            # Extract caption
            caption = sample.get('caption', '')
            aspect_list = sample.get('aspect_list', [])
            ytid = sample.get('ytid', f'track_{idx}')

            # Get audio
            if 'audio' not in sample or sample['audio'] is None:
                continue

            audio_data = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']

            # Detect BPM and key
            key, bpm = detect_key_bpm(audio_data, sample_rate)

            # Save audio file
            audio_filename = f"{ytid}_{idx:06d}.wav"
            audio_path = os.path.join(TRAIN_DIR, "audio_files", audio_filename)

            # Resample to 32kHz for MusicGen
            if sample_rate != 32000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=32000)
                sample_rate = 32000

            # Ensure stereo
            if len(audio_data.shape) == 1:
                audio_data = librosa.to_mono(audio_data)
                audio_data = np.stack([audio_data, audio_data])  # Duplicate to stereo

            sf.write(audio_path, audio_data.T, sample_rate)

            # Create metadata entry
            genre = ', '.join(aspect_list) if aspect_list else "Music"

            entry = {
                "path": f"audio_files/{audio_filename}",
                "bpm": bpm,
                "key": key,
                "genre": genre,
                "prompt": caption,
                "duration": len(audio_data) / sample_rate,
                "ytid": ytid
            }

            train_data.append(entry)

        except Exception as e:
            print(f"\n⚠️  Error processing sample {idx}: {e}")
            continue

    # Save training metadata
    train_jsonl = os.path.join(TRAIN_DIR, "data.jsonl")
    with open(train_jsonl, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Saved {len(train_data)} training samples to {train_jsonl}")

    # Process evaluation data
    print("\n🔄 Processing evaluation data...")
    eval_data = []

    for idx, sample in enumerate(tqdm(eval_samples)):
        try:
            caption = sample.get('caption', '')
            aspect_list = sample.get('aspect_list', [])
            ytid = sample.get('ytid', f'track_{idx}')

            if 'audio' not in sample or sample['audio'] is None:
                continue

            audio_data = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']

            key, bpm = detect_key_bpm(audio_data, sample_rate)

            audio_filename = f"{ytid}_{idx:06d}.wav"
            audio_path = os.path.join(EVAL_DIR, "audio_files", audio_filename)

            if sample_rate != 32000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=32000)
                sample_rate = 32000

            if len(audio_data.shape) == 1:
                audio_data = librosa.to_mono(audio_data)
                audio_data = np.stack([audio_data, audio_data])

            sf.write(audio_path, audio_data.T, sample_rate)

            genre = ', '.join(aspect_list) if aspect_list else "Music"

            entry = {
                "path": f"audio_files/{audio_filename}",
                "bpm": bpm,
                "key": key,
                "genre": genre,
                "prompt": caption,
                "duration": len(audio_data) / sample_rate,
                "ytid": ytid
            }

            eval_data.append(entry)

        except Exception as e:
            print(f"\n⚠️  Error processing eval sample {idx}: {e}")
            continue

    # Save evaluation metadata
    eval_jsonl = os.path.join(EVAL_DIR, "data.jsonl")
    with open(eval_jsonl, 'w') as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Saved {len(eval_data)} evaluation samples to {eval_jsonl}")

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
    import numpy as np
    download_and_prepare()
