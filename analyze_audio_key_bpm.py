#!/usr/bin/env python3
"""
Analyze audio files for BPM and musical key detection using Essentia
"""
import os
import essentia
import essentia.standard as es
from pathlib import Path

def analyze_audio(file_path):
    """
    Analyze audio file for BPM and key using Essentia
    """
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    print("=" * 60)

    try:
        # Load audio
        loader = es.MonoLoader(filename=str(file_path))
        audio = loader()

        # Get sample rate and duration
        audio_info = es.AudioLoader(filename=str(file_path))
        audio_stereo, sr, channels, _, _, _ = audio_info()
        duration = len(audio) / sr

        print(f"Duration: {duration:.2f}s | Sample Rate: {sr}Hz | Channels: {channels}")

        # ===== BPM DETECTION using RhythmExtractor2013 =====
        # This is Essentia's state-of-the-art BPM detection algorithm
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        print(f"\nBPM Detection (RhythmExtractor2013):")
        print(f"  Detected: {bpm:.1f} BPM")
        print(f"  Confidence: {beats_confidence:.3f}")
        print(f"  Number of beats: {len(beats)}")

        # Calculate alternative BPM interpretations
        print(f"  Half-time: {bpm/2:.1f} BPM")
        print(f"  Double-time: {bpm*2:.1f} BPM")

        # ===== KEY DETECTION using KeyExtractor =====
        # Essentia's key detection trained on electronic music
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        print(f"\nKey Detection (KeyExtractor):")
        print(f"  Detected: {key} {scale}")
        print(f"  Strength: {strength:.3f}")

        # Additional key analysis using HPCP (Harmonic Pitch Class Profile)
        # This gives us the pitch class distribution
        spectrum = es.Spectrum()
        w = es.Windowing(type='blackmanharris62')
        spectral_peaks = es.SpectralPeaks()
        hpcp = es.HPCP()

        # Process audio in frames
        hpcp_values = []
        for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
            frame_spectrum = spectrum(w(frame))
            frequencies, magnitudes = spectral_peaks(frame_spectrum)
            hpcp_frame = hpcp(frequencies, magnitudes)
            hpcp_values.append(hpcp_frame)

        # Average HPCP over time
        import numpy as np
        hpcp_avg = np.mean(hpcp_values, axis=0)

        # Get top 3 pitch classes
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        top_3_idx = np.argsort(hpcp_avg)[-3:][::-1]

        print(f"  Top pitch classes:")
        for idx in top_3_idx:
            print(f"    {pitch_classes[idx]}: {hpcp_avg[idx]:.3f}")

        return {
            'file': os.path.basename(file_path),
            'duration': duration,
            'bpm': bpm,
            'bpm_confidence': beats_confidence,
            'key': key,
            'scale': scale,
            'key_strength': strength
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    audio_dir = "/home/ubuntu/beats/unkown_model_beats"

    print("=" * 60)
    print("AUDIO KEY & BPM ANALYSIS (Essentia)")
    print("=" * 60)

    # Find all audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(Path(audio_dir).glob(ext))

    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"\nFound {len(audio_files)} audio files\n")

    results = []
    for audio_file in audio_files:
        result = analyze_audio(str(audio_file))
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'File':<30} {'BPM':<10} {'Conf':<8} {'Key':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['file']:<30} {r['bpm']:.1f}   {r['bpm_confidence']:.3f}  {r['key']} {r['scale']}")

if __name__ == "__main__":
    main()
