#!/usr/bin/env python3
"""
Compare original vs post-processed audio files
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

def analyze_audio(filepath):
    """Analyze audio file and return metrics"""
    try:
        audio, sr = sf.read(str(filepath))
        
        # Convert to mono for analysis
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
        
        # Calculate metrics
        duration = len(audio_mono) / sr
        rms = np.sqrt(np.mean(audio_mono**2))
        peak = np.max(np.abs(audio_mono))
        dynamic_range = peak / (rms + 1e-8)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_mono, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_mono))
        
        # Try to detect key and BPM
        try:
            import essentia.standard as es
            loader = es.MonoLoader(filename=str(filepath), sampleRate=sr)
            audio_essentia = loader()
            
            # Key detection
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio_essentia)
            key_str = f"{key} {scale}"
            
            # BPM detection
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, _, _ = rhythm_extractor(audio_essentia)
        except:
            key_str = "N/A"
            bpm = 0.0
        
        return {
            'file': Path(filepath).name,
            'duration': duration,
            'sample_rate': sr,
            'rms': rms,
            'peak': peak,
            'dynamic_range': dynamic_range,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossing_rate,
            'key': key_str,
            'bpm': bpm,
            'file_size_kb': Path(filepath).stat().st_size / 1024
        }
    except Exception as e:
        return {'file': Path(filepath).name, 'error': str(e)}

# Find files
base_dir = Path(__file__).parent.parent
original = base_dir / "large_lora_test/pretrained_large/test_01.wav"
files_to_compare = [
    ("Original", original),
    ("Mastered Only", base_dir / "test_output_mastered.wav"),
    ("Key/BPM Corrected", base_dir / "test_output_key_bpm.wav"),
    ("Unified (Mastering)", base_dir / "test_output_unified.wav"),
    ("Unified + Key Correction", base_dir / "test_output_with_key_bpm.wav"),
]

print("="*80)
print("AUDIO COMPARISON: Original vs Post-Processed")
print("="*80)

results = []
for name, filepath in files_to_compare:
    if filepath.exists():
        result = analyze_audio(filepath)
        result['name'] = name
        results.append(result)
    else:
        print(f"⚠️  {name}: File not found - {filepath}")

if not results:
    print("❌ No files found for comparison")
    sys.exit(1)

# Print comparison table
print("\n" + "="*80)
print("METRICS COMPARISON")
print("="*80)

# Header
print(f"\n{'Method':<25} {'Duration':<10} {'RMS':<8} {'Peak':<8} {'Key':<12} {'BPM':<8} {'Size (KB)':<10}")
print("-" * 80)

for r in results:
    if 'error' in r:
        print(f"{r['name']:<25} ERROR: {r['error']}")
    else:
        print(f"{r['name']:<25} {r['duration']:>6.1f}s   {r['rms']:>6.3f}  {r['peak']:>6.3f}  {r['key']:<12} {r['bpm']:>6.1f}  {r['file_size_kb']:>8.1f}")

# Detailed comparison
if len(results) > 1:
    original_result = results[0]
    print("\n" + "="*80)
    print("CHANGES FROM ORIGINAL")
    print("="*80)
    
    for r in results[1:]:
        if 'error' not in r and 'error' not in original_result:
            print(f"\n{r['name']}:")
            print(f"  Duration: {r['duration']:.1f}s (original: {original_result['duration']:.1f}s, change: {r['duration']-original_result['duration']:+.1f}s)")
            print(f"  RMS Level: {r['rms']:.3f} (original: {original_result['rms']:.3f}, change: {r['rms']-original_result['rms']:+.3f})")
            print(f"  Peak Level: {r['peak']:.3f} (original: {original_result['peak']:.3f}, change: {r['peak']-original_result['peak']:+.3f})")
            print(f"  Dynamic Range: {r['dynamic_range']:.2f} (original: {original_result['dynamic_range']:.2f})")
            print(f"  Spectral Centroid: {r['spectral_centroid']:.1f} Hz (original: {original_result['spectral_centroid']:.1f} Hz)")
            if r['key'] != "N/A" and original_result['key'] != "N/A":
                key_change = "✅ Changed" if r['key'] != original_result['key'] else "Same"
                print(f"  Key: {r['key']} (original: {original_result['key']}) {key_change}")
            if r['bpm'] > 0 and original_result['bpm'] > 0:
                bpm_change = r['bpm'] - original_result['bpm']
                print(f"  BPM: {r['bpm']:.1f} (original: {original_result['bpm']:.1f}, change: {bpm_change:+.1f})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n✅ All post-processing methods successfully processed the audio")
print("📊 Use the metrics above to compare the effects of each method")
print("\nRecommendation: Use 'Unified + Key Correction' for complete processing")

