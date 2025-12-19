#!/usr/bin/env python3
"""
Normalize synthetic audio files to WAV format for training
Matches the format used in successful training (32kHz, stereo, normalized)
"""

import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

def normalize_audio(input_path, output_path):
    """Normalize audio to 32kHz stereo WAV"""
    # Load audio
    wav, sr = torchaudio.load(str(input_path))

    # Convert to stereo if needed
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]  # Take first 2 channels

    # Resample to 32kHz if needed
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        wav = resampler(wav)

    # Normalize to [-1, 1]
    wav = wav / (wav.abs().max() + 1e-8)

    # Save as WAV
    torchaudio.save(str(output_path), wav, 32000)

def main():
    input_dir = Path("/home/ubuntu/beats/synthetic_audio_files")
    output_dir = Path("/home/ubuntu/beats/musicgen/training/audiocraft/egs/synthetic_audio_normalized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all MP3 files
    mp3_files = sorted(input_dir.glob("*.mp3"))

    print(f"Found {len(mp3_files)} MP3 files")
    print(f"Output directory: {output_dir}")
    print("Converting to 32kHz stereo WAV...")

    for idx, mp3_file in enumerate(tqdm(mp3_files)):
        try:
            output_file = output_dir / f"normalized_{idx:04d}_{mp3_file.stem}.wav"
            normalize_audio(mp3_file, output_file)
        except Exception as e:
            print(f"\nError processing {mp3_file.name}: {e}")
            continue

    print(f"\n✅ Normalized {len(list(output_dir.glob('*.wav')))} files to {output_dir}")

if __name__ == "__main__":
    main()
