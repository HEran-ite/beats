#!/usr/bin/env python3
"""
Example: Integrating MusicGen Post-Processing with Test Scripts

This example shows how to integrate the post-processing module
with your existing MusicGen test and generation scripts.
"""

import sys
sys.path.insert(0, '/home/ubuntu/beats/musicgen/training/audiocraft')

import torch
import torchaudio
from pathlib import Path
from audiocraft.models import MusicGen
from musicgen_unified_postprocessor import MusicGenPostProcessor, generate_and_postprocess_musicgen


def example_1_basic_postprocessing():
    """Example 1: Basic post-processing after generation"""
    print("\n" + "="*60)
    print("Example 1: Basic Post-Processing")
    print("="*60)
    
    # Load model
    model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
    model.set_generation_params(duration=10)
    
    # Generate audio
    prompt = "120 bpm, F# minor, dark trap beat"
    print(f"\n🎵 Generating: '{prompt}'")
    wav = model.generate([prompt])
    
    # Post-process with mastering only
    print("\n🎛️  Applying post-processing...")
    processor = MusicGenPostProcessor(sample_rate=32000)
    processed_wav = processor.process(
        audio=wav,
        apply_mastering=True,
        return_tensor=True
    )
    
    # Save
    output_path = "example_1_basic.wav"
    torchaudio.save(output_path, processed_wav[0].cpu(), 32000)
    print(f"✅ Saved: {output_path}")


def example_2_with_key_bpm_correction():
    """Example 2: Post-processing with key and BPM correction"""
    print("\n" + "="*60)
    print("Example 2: With Key/BPM Correction")
    print("="*60)
    
    # Load model
    model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
    model.set_generation_params(duration=10)
    
    # Generate audio
    prompt = "energetic hip-hop beat"
    print(f"\n🎵 Generating: '{prompt}'")
    wav = model.generate([prompt])
    
    # Post-process with key/BPM correction + mastering
    print("\n🎛️  Applying post-processing with corrections...")
    processor = MusicGenPostProcessor(sample_rate=32000)
    processed_wav = processor.process(
        audio=wav,
        apply_mastering=True,
        target_key="F#-",  # Force to F# minor
        target_bpm=140.0,  # Force to 140 BPM
        return_tensor=True
    )
    
    # Save
    output_path = "example_2_corrected.wav"
    torchaudio.save(output_path, processed_wav[0].cpu(), 32000)
    print(f"✅ Saved: {output_path}")


def example_3_convenience_function():
    """Example 3: Using the convenience function"""
    print("\n" + "="*60)
    print("Example 3: Convenience Function")
    print("="*60)
    
    # Load model
    model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
    
    # Generate and post-process in one step
    prompt = "cinematic orchestral music, 120 bpm"
    print(f"\n🎵 Generating and post-processing: '{prompt}'")
    
    processed_wav, sr = generate_and_postprocess_musicgen(
        model=model,
        prompt=prompt,
        duration=10.0,
        apply_mastering=True,
        target_key="C+",
        target_bpm=120.0,
        sample_rate=32000
    )
    
    # Save
    output_path = "example_3_convenience.wav"
    torchaudio.save(output_path, processed_wav[0].cpu(), sr)
    print(f"✅ Saved: {output_path}")


def example_4_integration_with_test_script():
    """Example 4: How to integrate with test_large_lora.py"""
    print("\n" + "="*60)
    print("Example 4: Integration Pattern for test_large_lora.py")
    print("="*60)
    
    print("""
# In your test_large_lora.py, modify the generate_music function:

def generate_music(model, prompt, duration=15, model_name="model", 
                   apply_postprocessing=True, target_key=None, target_bpm=None):
    \"\"\"Generate music with MusicGen, optionally with post-processing\"\"\"
    try:
        start_time = time.time()
        
        with torch.no_grad():
            wav = model.generate([prompt], progress=False)
        
        generation_time = time.time() - start_time
        
        # Apply post-processing if requested
        if apply_postprocessing:
            from musicgen_unified_postprocessor import MusicGenPostProcessor
            processor = MusicGenPostProcessor(sample_rate=model.sample_rate)
            wav = processor.process(
                audio=wav,
                apply_mastering=True,
                target_key=target_key,
                target_bpm=target_bpm,
                return_tensor=True
            )
        
        return wav, model.sample_rate, generation_time
    
    except Exception as e:
        print(f"      ❌ Generation error: {e}")
        return None, None, 0

# Then in your test loop:
wav, sr, gen_time = generate_music(
    model, 
    prompt,
    apply_postprocessing=True,
    target_key="F#-" if expected_key else None,
    target_bpm=expected_bpm if expected_bpm else None
)
""")


def example_5_process_existing_file():
    """Example 5: Post-process an existing audio file"""
    print("\n" + "="*60)
    print("Example 5: Process Existing File")
    print("="*60)
    
    # Assume you have a generated file
    input_file = "generated_audio.wav"
    
    if not Path(input_file).exists():
        print(f"⚠️  File not found: {input_file}")
        print("   (This is just an example - create a file first)")
        return
    
    # Process it
    processor = MusicGenPostProcessor(sample_rate=32000)
    output_path = processor.process_file(
        input_path=input_file,
        output_path="processed_audio.wav",
        apply_mastering=True,
        target_key="C+",
        target_bpm=120.0
    )
    
    print(f"✅ Processed: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MusicGen Post-Processing Integration Examples")
    print("="*60)
    
    # Run examples (comment out ones you don't want to run)
    try:
        example_1_basic_postprocessing()
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
    
    try:
        example_2_with_key_bpm_correction()
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
    
    try:
        example_3_convenience_function()
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
    
    example_4_integration_with_test_script()
    
    try:
        example_5_process_existing_file()
    except Exception as e:
        print(f"❌ Example 5 failed: {e}")
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


