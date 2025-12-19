#!/usr/bin/env python3
"""
Professional Audio Post-Processing Pipeline for Beats
Includes mastering chain: EQ, compression, limiting, normalization, and stereo enhancement
"""

import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


class AudioPostProcessor:
    """
    Professional beat mastering and post-processing pipeline
    """

    def __init__(self, sample_rate: int = 48000):
        """
        Initialize audio post-processor

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def high_pass_filter(
        self, audio: np.ndarray, cutoff_freq: float = 30, order: int = 4
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove DC offset and sub-bass rumble

        Args:
            audio: Audio array (channels x samples)
            cutoff_freq: High-pass cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered audio
        """
        sos = signal.butter(order, cutoff_freq, 'hp', fs=self.sample_rate, output='sos')

        if audio.ndim == 1:
            return signal.sosfilt(sos, audio)
        else:
            return signal.sosfilt(sos, audio, axis=-1)

    def low_pass_filter(
        self, audio: np.ndarray, cutoff_freq: float = 18000, order: int = 6
    ) -> np.ndarray:
        """
        Apply low-pass filter to remove harsh high frequencies

        Args:
            audio: Audio array (channels x samples)
            cutoff_freq: Low-pass cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered audio
        """
        sos = signal.butter(order, cutoff_freq, 'lp', fs=self.sample_rate, output='sos')

        if audio.ndim == 1:
            return signal.sosfilt(sos, audio)
        else:
            return signal.sosfilt(sos, audio, axis=-1)

    def parametric_eq(
        self,
        audio: np.ndarray,
        freq: float,
        gain_db: float,
        q: float = 1.0
    ) -> np.ndarray:
        """
        Apply parametric EQ at specific frequency

        Args:
            audio: Audio array
            freq: Center frequency in Hz
            gain_db: Gain in dB (positive for boost, negative for cut)
            q: Q factor (bandwidth)

        Returns:
            EQ'd audio
        """
        # Convert gain from dB
        gain_linear = 10 ** (gain_db / 20)

        # Design peaking filter
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)

        A = gain_linear

        # Peaking filter coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        if audio.ndim == 1:
            return signal.lfilter(b, a, audio)
        else:
            return signal.lfilter(b, a, audio, axis=-1)

    def compressor(
        self,
        audio: np.ndarray,
        threshold_db: float = -20,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        makeup_gain_db: float = 0.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression

        Args:
            audio: Audio array
            threshold_db: Compression threshold in dB
            ratio: Compression ratio (e.g., 4:1)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            makeup_gain_db: Makeup gain in dB

        Returns:
            Compressed audio
        """
        # Convert parameters
        threshold = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1 / (self.sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1 / (self.sample_rate * release_ms / 1000))
        makeup_gain = 10 ** (makeup_gain_db / 20)

        # Process each channel
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False

        output = np.zeros_like(audio)

        for ch in range(audio.shape[0]):
            channel = audio[ch]
            envelope = 0.0

            compressed = np.zeros_like(channel)

            for i, sample in enumerate(channel):
                # Get envelope
                abs_sample = abs(sample)
                if abs_sample > envelope:
                    envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_sample
                else:
                    envelope = release_coeff * envelope + (1 - release_coeff) * abs_sample

                # Apply compression
                if envelope > threshold:
                    # Calculate gain reduction
                    gain = threshold + (envelope - threshold) / ratio
                    gain = gain / (envelope + 1e-8)
                else:
                    gain = 1.0

                compressed[i] = sample * gain

            output[ch] = compressed * makeup_gain

        if squeeze_output:
            output = output[0]

        return output

    def soft_limiter(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Apply soft limiting using tanh to prevent clipping

        Args:
            audio: Audio array
            threshold: Limiting threshold (0-1)

        Returns:
            Limited audio
        """
        # Scale to threshold
        scale = threshold / (np.max(np.abs(audio)) + 1e-8)
        scale = min(scale, 1.0)

        # Apply soft clipping
        limited = np.tanh(audio * scale / threshold) * threshold

        return limited

    def loudness_normalize(
        self, audio: np.ndarray, target_lufs: float = -14.0
    ) -> np.ndarray:
        """
        Normalize audio to target LUFS (Loudness Units Full Scale)

        Args:
            audio: Audio array (channels x samples)
            target_lufs: Target LUFS level (e.g., -14 for streaming)

        Returns:
            Normalized audio
        """
        try:
            import pyloudnorm as pyln

            # Create meter
            meter = pyln.Meter(self.sample_rate)

            # Ensure correct shape for pyloudnorm (samples x channels)
            if audio.ndim == 1:
                audio_transposed = audio[:, np.newaxis]
            else:
                audio_transposed = audio.T

            # Measure loudness
            try:
                loudness = meter.integrated_loudness(audio_transposed)
            except ValueError:
                # If audio is too quiet, return as-is
                warnings.warn("Audio too quiet for loudness measurement, skipping normalization")
                return audio

            # Normalize
            normalized = pyln.normalize.loudness(audio_transposed, loudness, target_lufs)

            # Restore original shape
            if audio.ndim == 1:
                return normalized.flatten()
            else:
                return normalized.T

        except ImportError:
            warnings.warn("pyloudnorm not installed, using peak normalization instead")
            # Fallback to peak normalization
            peak = np.max(np.abs(audio))
            target_peak = 10 ** (target_lufs / 20)
            return audio * (target_peak / (peak + 1e-8))

    def stereo_widening(
        self, audio: np.ndarray, width: float = 1.2
    ) -> np.ndarray:
        """
        Apply stereo widening using mid-side processing

        Args:
            audio: Stereo audio array (2 x samples)
            width: Widening factor (1.0 = no change, >1.0 = wider)

        Returns:
            Widened stereo audio
        """
        if audio.shape[0] != 2:
            warnings.warn("Stereo widening requires stereo audio, skipping")
            return audio

        # Extract mid and side
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2

        # Widen side channel
        side_widened = side * width

        # Reconstruct stereo
        left = mid + side_widened
        right = mid - side_widened

        return np.array([left, right])

    def noise_gate(
        self,
        audio: np.ndarray,
        threshold_db: float = -60,
        attack_ms: float = 1.0,
        release_ms: float = 100.0
    ) -> np.ndarray:
        """
        Apply noise gate to reduce background noise

        Args:
            audio: Audio array
            threshold_db: Gate threshold in dB
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            Gated audio
        """
        threshold = 10 ** (threshold_db / 20)
        attack_coeff = np.exp(-1 / (self.sample_rate * attack_ms / 1000))
        release_coeff = np.exp(-1 / (self.sample_rate * release_ms / 1000))

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False

        output = np.zeros_like(audio)

        for ch in range(audio.shape[0]):
            channel = audio[ch]
            envelope = 0.0

            gated = np.zeros_like(channel)

            for i, sample in enumerate(channel):
                # Get envelope
                abs_sample = abs(sample)
                if abs_sample > envelope:
                    envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_sample
                else:
                    envelope = release_coeff * envelope + (1 - release_coeff) * abs_sample

                # Apply gate
                if envelope > threshold:
                    gain = 1.0
                else:
                    gain = 0.0

                gated[i] = sample * gain

            output[ch] = gated

        if squeeze_output:
            output = output[0]

        return output

    def denoise_spectral(
        self, audio: np.ndarray, noise_reduce_factor: float = 0.7
    ) -> np.ndarray:
        """
        Apply spectral noise reduction using noise profile from first 0.5 seconds
        Optimized for MusicGen quantization noise reduction

        Args:
            audio: Audio array
            noise_reduce_factor: Strength of noise reduction (0-1, default 0.7 for beats)

        Returns:
            Denoised audio
        """
        try:
            import noisereduce as nr

            # Use first 0.5 seconds as noise profile
            noise_duration = int(0.5 * self.sample_rate)

            if audio.ndim == 1:
                noise_sample = audio[:noise_duration]
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    stationary=True,
                    prop_decrease=noise_reduce_factor
                )
            else:
                # Process each channel
                reduced = np.zeros_like(audio)
                for ch in range(audio.shape[0]):
                    noise_sample = audio[ch, :noise_duration]
                    reduced[ch] = nr.reduce_noise(
                        y=audio[ch],
                        sr=self.sample_rate,
                        y_noise=noise_sample,
                        stationary=True,
                        prop_decrease=noise_reduce_factor
                    )

            return reduced

        except ImportError:
            warnings.warn("noisereduce not installed, skipping noise reduction")
            return audio

    def harmonic_exciter(
        self, audio: np.ndarray, freq_range: Tuple[float, float] = (8000, 16000), 
        amount: float = 0.15
    ) -> np.ndarray:
        """
        Add harmonic excitement to high frequencies (8-16kHz) to restore air and presence.
        This is critical for MusicGen outputs which are weak above 12-14kHz.
        
        Args:
            audio: Audio array (channels x samples)
            freq_range: Frequency range to enhance (Hz)
            amount: Amount of harmonic excitement (0-1)
            
        Returns:
            Enhanced audio with restored high-end
        """
        try:
            # Ensure we have enough bandwidth
            nyquist = self.sample_rate / 2
            if freq_range[1] > nyquist * 0.9:
                # Adjust range if too close to Nyquist
                freq_range = (freq_range[0], nyquist * 0.85)
            
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
                squeeze = True
            else:
                squeeze = False
            
            enhanced = np.zeros_like(audio)
            
            for ch in range(audio.shape[0]):
                channel = audio[ch]
                
                # Apply gentle high-shelf boost in the target range
                # Use parametric EQ with high Q for narrow boost
                center_freq = (freq_range[0] + freq_range[1]) / 2
                gain_db = 2.0 * amount  # Subtle boost (0-2dB)
                
                # Apply high-shelf filter
                enhanced[ch] = self.parametric_eq(
                    channel, 
                    freq=center_freq, 
                    gain_db=gain_db, 
                    q=0.7
                )
                
                # Add subtle harmonic distortion for "air"
                # Very light saturation in high frequencies only
                if amount > 0:
                    # Extract high frequencies
                    sos_high = signal.butter(4, freq_range[0] / nyquist, 'hp', output='sos')
                    high_freq = signal.sosfilt(sos_high, enhanced[ch])
                    
                    # Apply subtle soft saturation (tanh for smooth distortion)
                    saturation = np.tanh(high_freq * (1 + amount * 0.3))
                    
                    # Mix back (90% original, 10% saturated)
                    enhanced[ch] = enhanced[ch] * 0.9 + saturation * 0.1 * amount
            
            if squeeze:
                enhanced = enhanced[0]
            
            return enhanced
            
        except Exception as e:
            warnings.warn(f"Harmonic exciter failed: {e}, using original audio")
            return audio

    def process_beat(
        self,
        audio: np.ndarray,
        target_lufs: float = -14.0,
        apply_eq: bool = True,
        apply_compression: bool = True,
        apply_denoise: bool = True,  # ENABLED BY DEFAULT for MusicGen noise reduction
        apply_stereo_width: bool = True,
        apply_harmonic_exciter: bool = True  # NEW: High-end enhancement
    ) -> np.ndarray:
        """
        Complete mastering chain for beats with MusicGen noise reduction optimizations

        Args:
            audio: Audio array (channels x samples)
            target_lufs: Target LUFS level
            apply_eq: Whether to apply EQ
            apply_compression: Whether to apply compression
            apply_denoise: Whether to apply noise reduction (default: True for MusicGen)
            apply_stereo_width: Whether to apply stereo widening
            apply_harmonic_exciter: Whether to apply high-end enhancement (default: True)

        Returns:
            Processed audio
        """
        print("  🎛️  Applying professional mastering chain (MusicGen optimized)...")

        # 1. Noise gate (remove silence/noise)
        print("     1/8: Noise gate")
        audio = self.noise_gate(audio, threshold_db=-60)

        # 2. Spectral noise reduction (ENABLED by default for MusicGen quantization noise)
        if apply_denoise:
            print("     2/8: Spectral noise reduction (MusicGen quantization noise)")
            audio = self.denoise_spectral(audio, noise_reduce_factor=0.7)
        else:
            print("     2/8: Noise reduction (skipped)")

        # 3. High-pass filter (remove DC offset and rumble)
        print("     3/8: High-pass filter (30 Hz)")
        audio = self.high_pass_filter(audio, cutoff_freq=30)

        # 4. EQ adjustments (optional)
        if apply_eq:
            print("     4/8: EQ adjustments")
            # Boost sub-bass slightly for beats
            audio = self.parametric_eq(audio, freq=60, gain_db=1.5, q=0.7)
            # Boost presence for clarity
            audio = self.parametric_eq(audio, freq=3000, gain_db=1.0, q=1.5)
            # Gentle low-pass to remove harsh highs (ensure below Nyquist)
            nyquist = self.sample_rate / 2
            cutoff = min(18000, nyquist * 0.9)  # Use 90% of Nyquist as max
            audio = self.low_pass_filter(audio, cutoff_freq=cutoff)
        else:
            print("     4/8: EQ (skipped)")

        # 5. Compression (optional)
        if apply_compression:
            print("     5/8: Dynamic compression")
            audio = self.compressor(
                audio,
                threshold_db=-20,
                ratio=3.0,
                attack_ms=5.0,
                release_ms=50.0,
                makeup_gain_db=2.0
            )
        else:
            print("     5/8: Compression (skipped)")

        # 6. Harmonic exciter (NEW: Restore high-end air and presence)
        # MusicGen is weak above 12-14kHz, this restores professional sound
        if apply_harmonic_exciter:
            print("     6/8: Harmonic exciter (high-end enhancement 8-16kHz)")
            audio = self.harmonic_exciter(audio, freq_range=(8000, 16000), amount=0.15)
        else:
            print("     6/8: Harmonic exciter (skipped)")

        # 7. Loudness normalization
        print(f"     7/8: Loudness normalization ({target_lufs} LUFS)")
        audio = self.loudness_normalize(audio, target_lufs=target_lufs)

        # 8. Stereo widening (for stereo audio)
        if apply_stereo_width and audio.ndim == 2 and audio.shape[0] == 2:
            print("     8/8: Stereo widening")
            audio = self.stereo_widening(audio, width=1.15)
        else:
            print("     8/8: Stereo widening (skipped)")

        # Final soft limiting
        print("     Final: Soft limiting")
        audio = self.soft_limiter(audio, threshold=0.95)

        print("  ✅ Mastering complete!")

        return audio


def postprocess_audio_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_lufs: float = -14.0,
    apply_eq: bool = True,
    apply_compression: bool = True,
    apply_denoise: bool = False,
    apply_stereo_width: bool = True
) -> Path:
    """
    Post-process an audio file with professional mastering chain

    Args:
        input_path: Path to input audio file
        output_path: Path to output file (default: input_path with _processed suffix)
        target_lufs: Target LUFS level
        apply_eq: Whether to apply EQ
        apply_compression: Whether to apply compression
        apply_denoise: Whether to apply noise reduction
        apply_stereo_width: Whether to apply stereo widening

    Returns:
        Path to processed audio file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
    else:
        output_path = Path(output_path)

    print(f"📂 Loading: {input_path.name}")

    # Load audio
    audio, sample_rate = sf.read(input_path)

    # Convert to channels-first if needed
    if audio.ndim == 2:
        audio = audio.T  # (samples, channels) -> (channels, samples)

    # Create processor
    processor = AudioPostProcessor(sample_rate=sample_rate)

    # Process
    processed = processor.process_beat(
        audio,
        target_lufs=target_lufs,
        apply_eq=apply_eq,
        apply_compression=apply_compression,
        apply_denoise=apply_denoise,
        apply_stereo_width=apply_stereo_width
    )

    # Convert back to samples-first
    if processed.ndim == 2:
        processed = processed.T

    # Save
    print(f"💾 Saving: {output_path.name}")
    sf.write(output_path, processed, sample_rate)

    return output_path


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        print("="*60)
        print("AUDIO POST-PROCESSING")
        print("="*60)

        processed_path = postprocess_audio_file(
            input_path=input_file,
            output_path=output_file,
            target_lufs=-14.0,
            apply_eq=True,
            apply_compression=True,
            apply_denoise=False,  # Set to True if you have noisereduce installed
            apply_stereo_width=True
        )

        print(f"\n✅ Processing complete: {processed_path}")
    else:
        print("Usage: python audio_postprocessor.py <input_file> [output_file]")
        print("Example: python audio_postprocessor.py beat.wav beat_mastered.wav")


