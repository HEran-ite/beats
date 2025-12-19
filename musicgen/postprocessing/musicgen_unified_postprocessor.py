#!/usr/bin/env python3
"""
MusicGen Post-Processing Pipeline

Integrates professional audio mastering and key/BPM correction for MusicGen-generated audio.
This module provides a unified interface to enhance MusicGen outputs with:
- Professional mastering (EQ, compression, limiting, normalization)
- Key and BPM correction (pitch-shifting and time-stretching)
- Noise reduction and audio enhancement

Usage:
    from musicgen_postprocessor import MusicGenPostProcessor
    
    processor = MusicGenPostProcessor(sample_rate=32000)
    enhanced_audio = processor.process(
        audio_tensor,  # torch.Tensor from MusicGen
        apply_mastering=True,
        target_key="F#-",
        target_bpm=140.0
    )
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional, Dict
import warnings
import tempfile
import os

# Import local post-processors (prefer package-relative imports, fall back to local)
try:
    from .mastering_audio_postprocessor import AudioPostProcessor
    AUDIO_POSTPROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from mastering_audio_postprocessor import AudioPostProcessor
        AUDIO_POSTPROCESSOR_AVAILABLE = True
    except ImportError:
        AUDIO_POSTPROCESSOR_AVAILABLE = False
        warnings.warn(
            "AudioPostProcessor not available. Ensure mastering_audio_postprocessor.py is in the same directory."
        )

try:
    from .key_bpm_correction_postprocessor import KeyBPMPostProcessor
    KEY_BPM_POSTPROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from key_bpm_correction_postprocessor import KeyBPMPostProcessor
        KEY_BPM_POSTPROCESSOR_AVAILABLE = True
    except ImportError:
        KEY_BPM_POSTPROCESSOR_AVAILABLE = False
        warnings.warn(
            "KeyBPMPostProcessor not available. Ensure key_bpm_correction_postprocessor.py is in the same directory."
        )


class MusicGenPostProcessor:
    """
    Unified post-processing pipeline for MusicGen-generated audio
    
    Combines:
    - Professional mastering (EQ, compression, limiting, normalization)
    - Key and BPM correction (pitch-shifting, time-stretching)
    - Audio enhancement and noise reduction
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        target_lufs: float = -14.0,
        apply_eq: bool = True,
        apply_compression: bool = True,
        apply_denoise: bool = False,
        apply_stereo_width: bool = True
    ):
        """
        Initialize MusicGen post-processor
        
        Args:
            sample_rate: Audio sample rate (MusicGen default: 32000 Hz)
            target_lufs: Target loudness in LUFS (default: -14.0 for streaming)
            apply_eq: Whether to apply EQ adjustments
            apply_compression: Whether to apply dynamic compression
            apply_denoise: Whether to apply noise reduction
            apply_stereo_width: Whether to apply stereo widening
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.apply_eq = apply_eq
        self.apply_compression = apply_compression
        self.apply_denoise = apply_denoise
        self.apply_stereo_width = apply_stereo_width
        self.apply_harmonic_exciter = apply_harmonic_exciter
        
        # Initialize sub-processors
        if AUDIO_POSTPROCESSOR_AVAILABLE:
            self.audio_processor = AudioPostProcessor(sample_rate=sample_rate)
        else:
            self.audio_processor = None
            
        if KEY_BPM_POSTPROCESSOR_AVAILABLE:
            self.key_bpm_processor = KeyBPMPostProcessor(sample_rate=sample_rate)
        else:
            self.key_bpm_processor = None
    
    def _tensor_to_numpy(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert MusicGen audio tensor to numpy array
        
        MusicGen output format: (batch, channels, samples)
        Returns: (channels, samples) numpy array
        """
        # Handle batch dimension
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor[0]  # Take first batch item
        
        # Convert to CPU and numpy
        audio_np = audio_tensor.cpu().numpy()
        
        # Ensure shape is (channels, samples)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]  # Add channel dimension
        
        return audio_np
    
    def _numpy_to_tensor(self, audio_np: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array back to torch tensor
        
        Args:
            audio_np: (channels, samples) numpy array
            
        Returns:
            torch.Tensor: (1, channels, samples) tensor
        """
        # Ensure shape is (channels, samples)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_np).float()
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        return audio_tensor
    
    def _save_temp_audio(self, audio_np: np.ndarray, sample_rate: int) -> Path:
        """Save audio to temporary file for key/BPM processing"""
        # Convert to (samples, channels) for soundfile
        if audio_np.ndim == 2:
            audio_save = audio_np.T  # (channels, samples) -> (samples, channels)
        else:
            audio_save = audio_np
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        # Save
        sf.write(temp_path, audio_save, sample_rate)
        return Path(temp_path)
    
    def process(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, Path],
        apply_mastering: bool = False,  # Disabled by default - focus on key/BPM correction
        target_key: Optional[str] = None,
        target_bpm: Optional[float] = None,
        return_tensor: bool = True,
        **mastering_kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process MusicGen-generated audio with mastering and/or key/BPM correction
        
        Args:
            audio: Audio input - can be:
                - torch.Tensor: (batch, channels, samples) or (channels, samples)
                - np.ndarray: (channels, samples) or (samples, channels)
                - str/Path: Path to audio file
            apply_mastering: Whether to apply professional mastering chain
            target_key: Target key (e.g., "F#-", "C+") - if None, no key correction
            target_bpm: Target BPM (float) - if None, no BPM correction
            return_tensor: If True, return torch.Tensor; if False, return np.ndarray
            **mastering_kwargs: Additional kwargs for mastering (target_lufs, apply_eq, etc.)
            
        Returns:
            Processed audio as torch.Tensor or np.ndarray
        """
        if apply_mastering:
            print("🎛️  Starting MusicGen post-processing (mastering + key/BPM correction)...")
        else:
            print("🎛️  Starting MusicGen post-processing (key/BPM correction only, mastering disabled)...")
        
        # Load/convert audio to numpy
        if isinstance(audio, (str, Path)):
            audio_np, sr = sf.read(str(audio))
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            else:
                audio_np = audio_np.T  # (samples, channels) -> (channels, samples)
            if sr != self.sample_rate:
                import librosa
                audio_resampled = []
                for ch in audio_np:
                    resampled = librosa.resample(ch, orig_sr=sr, target_sr=self.sample_rate)
                    audio_resampled.append(resampled)
                audio_np = np.array(audio_resampled)
        elif isinstance(audio, torch.Tensor):
            audio_np = self._tensor_to_numpy(audio)
        else:
            # Assume numpy array
            audio_np = audio
            if audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
                # Likely (samples, channels), transpose
                audio_np = audio_np.T
        
        # Ensure shape is (channels, samples)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        
        # Step 1: Professional mastering FIRST (if enabled)
        if apply_mastering and self.audio_processor:
            has_key_bpm = (target_key or target_bpm) and self.key_bpm_processor
            step_msg = " (step 1/2)" if has_key_bpm else ""
            print(f"  🎚️  Applying professional mastering{step_msg}...")
            
            # Merge mastering kwargs with defaults
            # ENABLE denoising by default for MusicGen quantization noise reduction
            mastering_params = {
                'target_lufs': mastering_kwargs.get('target_lufs', self.target_lufs),
                'apply_eq': mastering_kwargs.get('apply_eq', self.apply_eq),
                'apply_compression': mastering_kwargs.get('apply_compression', self.apply_compression),
                'apply_denoise': mastering_kwargs.get('apply_denoise', True),  # ENABLED by default
                'apply_stereo_width': mastering_kwargs.get('apply_stereo_width', self.apply_stereo_width),
                'apply_harmonic_exciter': mastering_kwargs.get('apply_harmonic_exciter', True),  # ENABLED by default
            }
            
            # Apply mastering chain
            audio_np = self.audio_processor.process_beat(
                audio_np,
                **mastering_params
            )
            
            print("  ✅ Mastering complete!")
        
        # Step 2: Key/BPM correction (on raw or mastered audio)
        if (target_key or target_bpm) and self.key_bpm_processor:
            has_mastering = apply_mastering and self.audio_processor
            step_msg = " (step 2/2)" if has_mastering else ""
            print(f"  🎹 Applying key/BPM correction{step_msg}...")
            
            # Save to temp file for key/BPM processing
            temp_path = self._save_temp_audio(audio_np, self.sample_rate)
            
            try:
                # Process with key/BPM correction
                corrected_path, correction_info = self.key_bpm_processor.process_file(
                    temp_path,
                    target_key=target_key,
                    target_bpm=target_bpm,
                    auto_detect=True
                )
                
                # Load corrected audio
                audio_corrected, sr = sf.read(str(corrected_path))
                if audio_corrected.ndim == 1:
                    audio_corrected = audio_corrected[np.newaxis, :]
                else:
                    audio_corrected = audio_corrected.T  # (samples, channels) -> (channels, samples)
                
                audio_np = audio_corrected
                
                # Verify corrections were applied
                corrections = correction_info.get('corrections_applied', [])
                if 'key' in corrections:
                    print(f"     Key shift: {correction_info.get('key_shift', 0):+.1f} semitones")
                if 'bpm' in corrections:
                    print(f"     BPM stretch: {correction_info.get('bpm_stretch', 1.0):.2f}x")
                
                print(f"  ✅ Key/BPM correction complete: {corrections}")
                
            finally:
                # Cleanup temp files
                if temp_path.exists():
                    temp_path.unlink()
                if corrected_path.exists() and corrected_path != temp_path:
                    corrected_path.unlink()
        
        
        # Convert back to requested format
        if return_tensor:
            return self._numpy_to_tensor(audio_np)
        else:
            return audio_np
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        apply_mastering: bool = False,  # Disabled by default - focus on key/BPM correction
        target_key: Optional[str] = None,
        target_bpm: Optional[float] = None,
        **mastering_kwargs
    ) -> Path:
        """
        Process audio file with post-processing
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output file (default: input_path with _processed suffix)
            apply_mastering: Whether to apply mastering
            target_key: Target key for correction
            target_bpm: Target BPM for correction
            **mastering_kwargs: Additional mastering parameters
            
        Returns:
            Path to processed audio file
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # Process audio
        processed_audio = self.process(
            audio=input_path,
            apply_mastering=apply_mastering,
            target_key=target_key,
            target_bpm=target_bpm,
            return_tensor=False,
            **mastering_kwargs
        )
        
        # Save processed audio
        # Convert to (samples, channels) for soundfile
        if processed_audio.ndim == 2:
            audio_save = processed_audio.T  # (channels, samples) -> (samples, channels)
        else:
            audio_save = processed_audio
        
        sf.write(str(output_path), audio_save, self.sample_rate)
        
        print(f"💾 Saved processed audio: {output_path}")
        
        return output_path


def generate_and_postprocess_musicgen(
    model,
    prompt: str,
    duration: float = 15.0,
    apply_mastering: bool = False,  # Disabled by default - focus on key/BPM correction
    target_key: Optional[str] = None,
    target_bpm: Optional[float] = None,
    sample_rate: int = 32000,
    **postprocessing_kwargs
) -> Tuple[torch.Tensor, int]:
    """
    Convenience function: Generate with MusicGen and apply post-processing
    
    Args:
        model: MusicGen model instance
        prompt: Text prompt for generation
        duration: Generation duration in seconds
        apply_mastering: Whether to apply mastering
        target_key: Target key for correction
        target_bpm: Target BPM for correction
        sample_rate: Audio sample rate
        **postprocessing_kwargs: Additional post-processing parameters
        
    Returns:
        (processed_audio_tensor, sample_rate)
    """
    print(f"🎵 Generating music with prompt: '{prompt}'")
    
    # Generate audio
    with torch.no_grad():
        wav = model.generate([prompt], progress=False)
    
    print(f"✅ Generated audio shape: {wav.shape}")
    
    # Apply post-processing
    processor = MusicGenPostProcessor(sample_rate=sample_rate)
    processed_wav = processor.process(
        audio=wav,
        apply_mastering=apply_mastering,
        target_key=target_key,
        target_bpm=target_bpm,
        return_tensor=True,
        **postprocessing_kwargs
    )
    
    return processed_wav, sample_rate


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Post-process MusicGen audio with mastering and key/BPM correction',
        prog='musicgen_unified_postprocessor.py'
    )
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('--output', help='Output path (default: input_processed.wav)')
    parser.add_argument('--target-key', help='Target key (e.g., F#-)')
    parser.add_argument('--target-bpm', type=float, help='Target BPM')
    parser.add_argument('--no-mastering', action='store_true', help='Skip mastering')
    parser.add_argument('--target-lufs', type=float, default=-14.0, help='Target LUFS')
    
    args = parser.parse_args()
    
    processor = MusicGenPostProcessor()
    
    input_path = Path(args.input)
    if input_path.is_file():
        output_path = processor.process_file(
            input_path,
            output_path=args.output,
            apply_mastering=not args.no_mastering,
            target_key=args.target_key,
            target_bpm=args.target_bpm,
            target_lufs=args.target_lufs
        )
        print(f"\n✅ Processing complete: {output_path}")
    else:
        print(f"❌ Input must be a file: {input_path}")

