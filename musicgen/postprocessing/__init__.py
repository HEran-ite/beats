"""
MusicGen Post-Processing Package

This package provides unified post-processing for MusicGen-generated audio.

Main module: musicgen_unified_postprocessor
- Combines professional mastering and key/BPM correction
- Use this for all post-processing needs

Dependencies:
- mastering_audio_postprocessor: Professional mastering chain
- key_bpm_correction_postprocessor: Key and BPM correction
"""

from .musicgen_unified_postprocessor import MusicGenPostProcessor, generate_and_postprocess_musicgen

__all__ = ['MusicGenPostProcessor', 'generate_and_postprocess_musicgen']

