# MusicGen Post-Processing Package

All post-processing modules are organized in this folder.

## Main File (Use This!)

**`musicgen_unified_postprocessor.py`** - **USE THIS ONE!**
- Combines everything: mastering + key/BPM correction
- Works with MusicGen outputs directly
- Handles torch tensors, numpy arrays, and file paths

```python
from postprocessing.musicgen_unified_postprocessor import MusicGenPostProcessor

processor = MusicGenPostProcessor(sample_rate=32000)
processed = processor.process(audio, apply_mastering=True, target_key="F#-", target_bpm=120.0)
```

## Supporting Files (Dependencies)

- `mastering_audio_postprocessor.py` - Professional mastering (EQ, compression, etc.)
- `key_bpm_correction_postprocessor.py` - Key and BPM correction

These are used by the unified processor - you don't need to import them directly.

## Other Files

- `test_postprocessing.py` - Test suite for all modules
- `example_postprocess_integration.py` - Usage examples
- `POST_PROCESSING_README.md` - Detailed documentation

## Quick Start

```python
# Import the unified processor
from postprocessing.musicgen_unified_postprocessor import MusicGenPostProcessor

# Create processor
processor = MusicGenPostProcessor(sample_rate=32000)

# Process audio (does everything!)
processed = processor.process(
    audio=wav_tensor,
    apply_mastering=True,
    target_key="F#-",
    target_bpm=120.0
)
```

