#!/usr/bin/env python3
"""
Test script for MusicGen post-processing modules

Tests:
1. Mastering audio post-processor
2. Key/BPM correction post-processor
3. Unified MusicGen post-processor
"""

import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("MUSICGEN POST-PROCESSING TEST SUITE")
print("="*80)

# Test 1: Import modules
print("\n" + "="*80)
print("TEST 1: Module Imports")
print("="*80)

try:
    from mastering_audio_postprocessor import AudioPostProcessor
    print("✅ Successfully imported AudioPostProcessor")
    MASTERING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import AudioPostProcessor: {e}")
    MASTERING_AVAILABLE = False

try:
    from key_bpm_correction_postprocessor import KeyBPMPostProcessor
    print("✅ Successfully imported KeyBPMPostProcessor")
    KEY_BPM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import KeyBPMPostProcessor: {e}")
    KEY_BPM_AVAILABLE = False

try:
    from musicgen_unified_postprocessor import MusicGenPostProcessor, generate_and_postprocess_musicgen
    print("✅ Successfully imported MusicGenPostProcessor")
    UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import MusicGenPostProcessor: {e}")
    UNIFIED_AVAILABLE = False

if not (MASTERING_AVAILABLE and KEY_BPM_AVAILABLE and UNIFIED_AVAILABLE):
    print("\n❌ Some modules failed to import. Cannot continue tests.")
    sys.exit(1)

# Test 2: Find test audio file
print("\n" + "="*80)
print("TEST 2: Finding Test Audio File")
print("="*80)

test_audio_paths = [
    "large_lora_test/pretrained_large/test_01.wav",
    "large_lora_test_all/pretrained_large/test_01.wav",
    "test_outputs/test_01.wav",
    "audiocraft/assets/epic.wav"
]

test_audio = None
for path in test_audio_paths:
    full_path = Path(__file__).parent / path
    if full_path.exists():
        test_audio = full_path
        print(f"✅ Found test audio: {test_audio}")
        break

if test_audio is None:
    print("⚠️  No test audio file found. Creating synthetic test audio...")
    # Create a simple test tone
    sample_rate = 32000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple tone with some harmonics
    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
    audio += np.sin(2 * np.pi * 880 * t) * 0.1  # A5 harmonic
    # Make it stereo
    audio_stereo = np.array([audio, audio])
    test_audio = Path(__file__).parent / "test_synthetic_audio.wav"
    sf.write(str(test_audio), audio_stereo.T, sample_rate)
    print(f"✅ Created synthetic test audio: {test_audio}")

# Test 3: Test Mastering Post-Processor
print("\n" + "="*80)
print("TEST 3: Mastering Audio Post-Processor")
print("="*80)

try:
    # Load audio
    audio, sr = sf.read(str(test_audio))
    print(f"   Loaded audio: {audio.shape}, sample_rate={sr}")
    
    # Convert to channels-first if needed
    if audio.ndim == 2:
        audio_channels_first = audio.T
    else:
        audio_channels_first = audio[np.newaxis, :]
    
    # Create processor
    processor = AudioPostProcessor(sample_rate=sr)
    print(f"   Created AudioPostProcessor with sample_rate={sr}")
    
    # Test individual functions
    print("   Testing high-pass filter...")
    filtered = processor.high_pass_filter(audio_channels_first, cutoff_freq=30)
    print(f"      ✅ High-pass filter: {filtered.shape}")
    
    print("   Testing parametric EQ...")
    eq_audio = processor.parametric_eq(audio_channels_first, freq=1000, gain_db=2.0, q=1.0)
    print(f"      ✅ Parametric EQ: {eq_audio.shape}")
    
    print("   Testing full mastering chain...")
    mastered = processor.process_beat(
        audio_channels_first,
        target_lufs=-14.0,
        apply_eq=True,
        apply_compression=True,
        apply_denoise=False,
        apply_stereo_width=True
    )
    print(f"      ✅ Full mastering chain: {mastered.shape}")
    
    # Save result
    output_path = Path(__file__).parent / "test_output_mastered.wav"
    if mastered.ndim == 2:
        mastered_save = mastered.T
    else:
        mastered_save = mastered
    sf.write(str(output_path), mastered_save, sr)
    print(f"   ✅ Saved mastered audio: {output_path}")
    
    print("✅ Mastering post-processor test PASSED")
    
except Exception as e:
    print(f"❌ Mastering post-processor test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Key/BPM Post-Processor
print("\n" + "="*80)
print("TEST 4: Key/BPM Correction Post-Processor")
print("="*80)

try:
    processor = KeyBPMPostProcessor(sample_rate=32000)
    print(f"   Created KeyBPMPostProcessor with sample_rate=32000")
    
    # Test key detection
    print("   Testing key detection...")
    detected_key, confidence = processor.detect_key(test_audio)
    print(f"      ✅ Detected key: {detected_key} (confidence: {confidence:.2%})")
    
    # Test BPM detection
    print("   Testing BPM detection...")
    detected_bpm = processor.detect_bpm(test_audio)
    print(f"      ✅ Detected BPM: {detected_bpm:.1f}")
    
    # Test key/BPM correction (with small correction to avoid major changes)
    print("   Testing key/BPM correction...")
    output_path, correction_info = processor.process_file(
        test_audio,
        target_key=None,  # Don't correct key for test
        target_bpm=None,  # Don't correct BPM for test
        output_path=Path(__file__).parent / "test_output_key_bpm.wav"
    )
    print(f"      ✅ Key/BPM processing complete: {output_path}")
    print(f"      Correction info: {correction_info}")
    
    print("✅ Key/BPM post-processor test PASSED")
    
except Exception as e:
    print(f"❌ Key/BPM post-processor test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test Unified Post-Processor
print("\n" + "="*80)
print("TEST 5: Unified MusicGen Post-Processor")
print("="*80)

try:
    # Test with file path
    print("   Testing with file path input...")
    processor = MusicGenPostProcessor(sample_rate=32000)
    
    output_path = processor.process_file(
        test_audio,
        output_path=Path(__file__).parent / "test_output_unified.wav",
        apply_mastering=True,
        target_key=None,
        target_bpm=None
    )
    print(f"      ✅ Processed file: {output_path}")
    
    # Test with numpy array
    print("   Testing with numpy array input...")
    audio, sr = sf.read(str(test_audio))
    if audio.ndim == 2:
        audio_np = audio.T  # Convert to channels-first
    else:
        audio_np = audio[np.newaxis, :]
    
    processed_np = processor.process(
        audio=audio_np,
        apply_mastering=True,
        return_tensor=False
    )
    print(f"      ✅ Processed numpy array: {processed_np.shape}")
    
    # Test with torch tensor
    print("   Testing with torch tensor input...")
    audio_tensor = torch.from_numpy(audio_np).float()
    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
    
    processed_tensor = processor.process(
        audio=audio_tensor,
        apply_mastering=True,
        return_tensor=True
    )
    print(f"      ✅ Processed torch tensor: {processed_tensor.shape}")
    
    # Save tensor result
    output_tensor_path = Path(__file__).parent / "test_output_unified_tensor.wav"
    torchaudio.save(str(output_tensor_path), processed_tensor[0].cpu(), sr)
    print(f"      ✅ Saved tensor result: {output_tensor_path}")
    
    print("✅ Unified post-processor test PASSED")
    
except Exception as e:
    print(f"❌ Unified post-processor test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test with Key/BPM Correction
print("\n" + "="*80)
print("TEST 6: Unified Post-Processor with Key/BPM Correction")
print("="*80)

try:
    processor = MusicGenPostProcessor(sample_rate=32000)
    
    # Get detected key/BPM first
    key_bpm_processor = KeyBPMPostProcessor(sample_rate=32000)
    detected_key, _ = key_bpm_processor.detect_key(test_audio)
    detected_bpm = key_bpm_processor.detect_bpm(test_audio)
    
    print(f"   Original: Key={detected_key}, BPM={detected_bpm:.1f}")
    
    # Test with key correction (shift by 2 semitones)
    target_key = "D+"  # Example target
    print(f"   Testing with target_key={target_key}...")
    
    output_path = processor.process_file(
        test_audio,
        output_path=Path(__file__).parent / "test_output_with_key_bpm.wav",
        apply_mastering=True,
        target_key=target_key,
        target_bpm=None  # Don't change BPM for this test
    )
    print(f"      ✅ Processed with key correction: {output_path}")
    
    print("✅ Key/BPM correction test PASSED")
    
except Exception as e:
    print(f"❌ Key/BPM correction test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

output_files = [
    "test_output_mastered.wav",
    "test_output_key_bpm.wav",
    "test_output_unified.wav",
    "test_output_unified_tensor.wav",
    "test_output_with_key_bpm.wav"
]

print("\nGenerated output files:")
for filename in output_files:
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"  ✅ {filename} ({size_kb:.1f} KB)")
    else:
        print(f"  ⚠️  {filename} (not found)")

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)

