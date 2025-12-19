#!/usr/bin/env python3
"""
Modified MusicGen sample script for Flask integration
Generates audio based on command line arguments and returns JSON output
"""

import os
# Disable xformers to avoid compatibility issues
os.environ["XFORMERS_DISABLED"] = "1"

import sys
# Add audiocraft paths if not in venv
audiocraft_paths = [
    "/home/ubuntu/beats/musicgen/training/audiocraft",
    "/home/ubuntu/beats/musicgen/audiocraft"
]
for path in audiocraft_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import torch
import torchaudio
# Import audiocraft first to ensure __version__ is set before importing models
import audiocraft
# Ensure __version__ exists (fallback if not set)
if not hasattr(audiocraft, '__version__'):
    audiocraft.__version__ = '1.4.0a2'
from audiocraft.models import MusicGen
import argparse
import json
from datetime import datetime

def generate_music_api(prompt, output_dir="generated_music", duration=15, model_size="medium", beat_name=None):
    """
    Generate music from a text prompt using MusicGen - API version
    Returns JSON with result information
    """
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Map model_size to actual model ID if needed
        # Medium uses stereo-melody-large pretrained model
        if model_size == "medium":
            model_id = "facebook/musicgen-stereo-melody-large"
        elif model_size == "large":
            model_id = "facebook/musicgen-large"
        elif model_size == "small":
            model_id = "facebook/musicgen-small"
        else:
            # Assume it's already a full model ID
            model_id = model_size
        
        # Print status for Flask to capture
        print(json.dumps({"status": "loading_model", "message": f"Loading MusicGen {model_id} model..."}))
        sys.stdout.flush()
        
        # Load the model
        model = MusicGen.get_pretrained(model_id)

        # Set generation parameters optimized for quality
        model.set_generation_params(
            duration=duration,
            top_k=250,              # Limit sampling to top 250 tokens for coherence
            top_p=0.0,              # Disable nucleus sampling (top_k is sufficient for music)
            temperature=1.0,        # Standard temperature for diversity
            cfg_coef=3.0,           # Balanced CFG for quality and coherence
            two_step_cfg=False,     # Disable two-step CFG for better quality
            use_sampling=True       # Enable sampling for more natural output
        )

        print(json.dumps({"status": "generating", "message": f"Generating music for: '{prompt}'"}))
        sys.stdout.flush()

        # Generate music
        descriptions = [prompt]
        wav = model.generate(descriptions, progress=True)
        
        # Create filename - use beat_name if provided, otherwise use prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if beat_name:
            # Sanitize beat name for filename
            safe_name = "".join(c for c in beat_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')[:50]
            filename = f"{safe_name}_{timestamp}.wav"
        else:
            # Fallback to prompt-based filename
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit filename length
        filename = f"{safe_prompt}_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save the audio file
        sample_rate = model.sample_rate
        torchaudio.save(filepath, wav[0].cpu(), sample_rate)
        
        # Return success JSON
        result = {
            "status": "success",
            "message": "Audio generated successfully",
            "filename": filename,
            "filepath": filepath,
            "duration": wav.shape[2]/sample_rate,
            "sample_rate": sample_rate,
            "audio_shape": list(wav.shape)
        }
        
        print(json.dumps(result))
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Error generating music: {str(e)}",
            "error": str(e)
        }
        print(json.dumps(error_result))
        return error_result

def main():
    parser = argparse.ArgumentParser(description='Generate music with MusicGen')
    parser.add_argument('--prompt', required=True, help='Text prompt for music generation')
    parser.add_argument('--duration', type=int, default=15, help='Duration in seconds (default: 15)')
    parser.add_argument('--model_size', choices=['small', 'medium', 'large'], default='medium', help='Model size (default: medium)')
    parser.add_argument('--output_dir', default='generated_music', help='Output directory (default: generated_music)')
    
    args = parser.parse_args()
    
    # Generate music
    result = generate_music_api(
        prompt=args.prompt,
        duration=args.duration,
        model_size=args.model_size,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)

if __name__ == "__main__":
    main()
