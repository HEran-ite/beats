# MusicGen Post-Processing Module

## Overview

The `musicgen_unified_postprocessor.py` module provides a unified interface for enhancing MusicGen-generated audio with professional mastering and key/BPM correction.

## Features

### 1. Professional Mastering Chain
- **Noise Gate**: Removes silence and background noise
- **Spectral Noise Reduction**: Reduces unwanted artifacts
- **High/Low-Pass Filtering**: Removes DC offset and harsh frequencies
- **Parametric EQ**: Frequency-specific adjustments
- **Dynamic Compression**: Controls dynamic range
- **Loudness Normalization**: EBU R128 standard (LUFS)
- **Stereo Widening**: Enhances stereo image
- **Soft Limiting**: Prevents clipping

### 2. Key and BPM Correction
- **Key Detection**: Automatic key detection using Essentia or librosa
- **Pitch-Shifting**: Corrects key without changing tempo
- **BPM Detection**: Automatic tempo detection
- **Time-Stretching**: Corrects tempo without changing pitch

## Dependencies

### Required
- `numpy`
- `torch`
- `torchaudio`
- `soundfile`
- `scipy`
- `librosa`

### Optional (for enhanced features)
- `pyloudnorm` (for accurate LUFS normalization)
- `noisereduce` (for spectral noise reduction)
- `essentia` (for better key/BPM detection)

### Local Modules
The module imports from the same directory:
- `mastering_audio_postprocessor.py` (AudioPostProcessor)
- `key_bpm_correction_postprocessor.py` (KeyBPMPostProcessor)

## Usage

### Basic Usage: Post-Process Generated Audio

```python
from audiocraft.models import MusicGen
from musicgen_unified_postprocessor import MusicGenPostProcessor

# Load model
model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
model.set_generation_params(duration=15)

# Generate audio
prompt = "120 bpm, F# minor, dark trap beat"
wav = model.generate([prompt])

# Post-process
processor = MusicGenPostProcessor(sample_rate=32000)
processed_wav = processor.process(
    audio=wav,
    apply_mastering=True,
    target_key="F#-",
    target_bpm=120.0
)

# Save
torchaudio.save("output.wav", processed_wav[0].cpu(), 32000)
```

### Using the Convenience Function

```python
from audiocraft.models import MusicGen
from musicgen_unified_postprocessor import generate_and_postprocess_musicgen

# Load model
model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")

# Generate and post-process in one step
processed_wav, sr = generate_and_postprocess_musicgen(
    model=model,
    prompt="120 bpm, F# minor, dark trap beat",
    duration=15.0,
    apply_mastering=True,
    target_key="F#-",
    target_bpm=120.0
)

# Save
torchaudio.save("output.wav", processed_wav[0].cpu(), sr)
```

### Post-Process Existing Audio Files

```python
from musicgen_unified_postprocessor import MusicGenPostProcessor

processor = MusicGenPostProcessor(sample_rate=32000)

# Process a single file
output_path = processor.process_file(
    input_path="generated_audio.wav",
    output_path="processed_audio.wav",
    apply_mastering=True,
    target_key="C+",
    target_bpm=140.0
)
```

### Command Line Usage

```bash
# Basic post-processing with mastering
python musicgen_unified_postprocessor.py input.wav --output output.wav

# With key and BPM correction
python musicgen_unified_postprocessor.py input.wav \
    --output output.wav \
    --target-key "F#-" \
    --target-bpm 120.0

# Skip mastering, only key/BPM correction
python musicgen_unified_postprocessor.py input.wav \
    --no-mastering \
    --target-key "C+" \
    --target-bpm 140.0

# Custom LUFS target
python musicgen_unified_postprocessor.py input.wav \
    --target-lufs -16.0
```

## Integration with Test Scripts

### Modify `test_large_lora.py`

```python
from musicgen_unified_postprocessor import MusicGenPostProcessor

# After generation
wav, sample_rate, generation_time = generate_music(model, prompt, duration)

# Add post-processing
processor = MusicGenPostProcessor(sample_rate=sample_rate)
processed_wav = processor.process(
    audio=wav,
    apply_mastering=True,
    target_key=expected_key.replace(" major", "+").replace(" minor", "-"),
    target_bpm=expected_bpm
)

# Use processed_wav instead of wav for saving/analysis
```

## Configuration Options

### Mastering Parameters

```python
processor = MusicGenPostProcessor(
    sample_rate=32000,
    target_lufs=-14.0,        # Target loudness (default: -14.0 for streaming)
    apply_eq=True,            # Apply EQ adjustments
    apply_compression=True,    # Apply dynamic compression
    apply_denoise=False,       # Apply noise reduction (requires noisereduce)
    apply_stereo_width=True   # Apply stereo widening
)
```

### Processing Options

```python
processed_audio = processor.process(
    audio=wav,
    apply_mastering=True,      # Enable/disable mastering
    target_key="F#-",          # Target key (None = no correction)
    target_bpm=120.0,          # Target BPM (None = no correction)
    return_tensor=True,         # Return torch.Tensor or np.ndarray
    target_lufs=-14.0,         # Override default LUFS
    apply_eq=True,             # Override default EQ setting
    # ... other mastering parameters
)
```

## Key Format

Keys should be specified in the format:
- Major keys: `"C+"`, `"D+"`, `"F#+"`, etc.
- Minor keys: `"C-"`, `"D-"`, `"F#-"`, etc.

The module will automatically detect the current key and apply pitch-shifting to match the target.

## Performance Considerations

- **Mastering**: Adds ~1-2 seconds processing time per audio file
- **Key/BPM Correction**: Adds ~2-5 seconds (includes detection + processing)
- **Memory**: Minimal additional memory usage (processes in chunks)

## Troubleshooting

### Import Errors

If you get import errors for `AudioPostProcessor` or `KeyBPMPostProcessor`:

1. Ensure all files are in the same directory:
   - `musicgen_unified_postprocessor.py`
   - `mastering_audio_postprocessor.py`
   - `key_bpm_correction_postprocessor.py`

2. Verify the files exist:
   ```bash
   ls /home/ubuntu/beats/musicgen/training/mastering_audio_postprocessor.py
   ls /home/ubuntu/beats/musicgen/training/key_bpm_correction_postprocessor.py
   ls /home/ubuntu/beats/musicgen/training/musicgen_unified_postprocessor.py
   ```

### Missing Dependencies

Install missing dependencies:
```bash
pip install pyloudnorm noisereduce essentia
```

### Essentia Not Available

The module will fall back to librosa for key/BPM detection if Essentia is not available. This is less accurate but still functional.

## Best Practices

1. **Always apply mastering** for production-ready audio
2. **Use key/BPM correction** when you need exact matches to specifications
3. **Test without post-processing first** to see raw model output
4. **Adjust LUFS** based on your distribution platform:
   - Spotify: -14 LUFS
   - YouTube: -14 LUFS
   - Apple Music: -16 LUFS
   - SoundCloud: -12 to -14 LUFS

## Example Workflow

```python
# 1. Generate with MusicGen
model = MusicGen.get_pretrained("facebook/musicgen-stereo-melody-large")
wav = model.generate(["120 bpm trap beat"])

# 2. Post-process
processor = MusicGenPostProcessor(sample_rate=32000)
processed = processor.process(
    wav,
    apply_mastering=True,
    target_key="F#-",
    target_bpm=120.0
)

# 3. Save
torchaudio.save("final_output.wav", processed[0].cpu(), 32000)

# 4. Verify
# Check key/BPM with audio analysis tools
```

## Future Enhancements

- [ ] Multi-band diffusion integration
- [ ] Real-time processing support
- [ ] Batch processing for multiple files
- [ ] Custom EQ presets per genre
- [ ] Automatic genre-specific mastering
- [ ] Integration with MusicGen training pipeline


