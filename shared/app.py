#!/usr/bin/env python3
"""
Flask backend for Music Generation UI
Supports MusicGen (local models)
"""

import os
# Disable xformers to avoid compatibility issues with PyTorch 2.9.0
os.environ["XFORMERS_DISABLED"] = "1"

from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
from urllib.parse import quote, unquote
import subprocess
import json
import threading
import time
import uuid
from datetime import datetime
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import gc
import base64
import io
import secrets

import sys


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_folder='static',
            static_url_path='/beats/static',
            template_folder='templates')

# URL prefix used by the frontend (and optional reverse proxy setups)
URL_PREFIX = os.environ.get("BEATS_URL_PREFIX", "/beats").rstrip("/") or ""

# Session/auth configuration (simple shared password)
# - Set BEATS_PASSWORD to the desired password.
# - Set FLASK_SECRET_KEY to a stable random value so sessions survive restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
BEATS_PASSWORD = os.environ.get("BEATS_PASSWORD", "").strip() or "beatworldhq"

# Enable template auto-reload so HTML changes are picked up immediately
app.config['TEMPLATES_AUTO_RELOAD'] = True

AUTH_SESSION_KEY = "beats_authenticated"


def _is_authenticated() -> bool:
    return bool(session.get(AUTH_SESSION_KEY))


def _is_api_request(path: str) -> bool:
    return path.startswith("/api/") or (URL_PREFIX and path.startswith(f"{URL_PREFIX}/api/"))


def _external_prefix() -> str:
    """
    Determine the externally-visible URL prefix.

    When running behind Nginx, requests to /beats/... are often rewritten to /...
    before reaching Flask, but Nginx forwards SCRIPT_NAME / X-Script-Name headers.
    """
    prefix = (
        (request.headers.get("X-Script-Name") or "").strip()
        or (request.headers.get("SCRIPT_NAME") or "").strip()
        or URL_PREFIX
    )
    prefix = prefix.rstrip("/")
    return prefix or ""


def _login_path() -> str:
    """
    Prefer /beats/login when URL_PREFIX is set, so the user stays within the app path.
    """
    prefix = _external_prefix()
    return f"{prefix}/login" if prefix else "/login"

def _external_path(path: str) -> str:
    """Map an internal Flask path to the external path (respecting prefix)."""
    prefix = _external_prefix()
    if not prefix:
        return path
    if path == "/":
        return f"{prefix}/"
    return f"{prefix}{path}"


def _safe_next_url(next_url: str | None) -> str:
    """
    Only allow internal redirects.
    Accepts:
      - /something
      - /beats/something (when prefixed)
    """
    if not next_url:
        return _external_path("/")
    candidate = next_url.strip()
    if not candidate.startswith("/"):
        return _external_path("/")
    # If we have an external prefix, force redirects to stay under it.
    prefix = _external_prefix()
    if prefix and not candidate.startswith(prefix):
        return _external_path("/")
    return candidate


@app.before_request
def _require_login():
    """
    Require a password for the UI + API.
    - HTML requests: redirect to login page
    - API requests: return 401 JSON (so frontend can redirect)
    """
    path = request.path or "/"

    # Allow static assets always
    if path.startswith("/beats/static") or path.startswith("/static/"):
        return None

    # Allow login/logout endpoints
    if path in ("/login", "/logout") or (URL_PREFIX and path in (f"{URL_PREFIX}/login", f"{URL_PREFIX}/logout")):
        return None

    # Already authenticated
    if _is_authenticated():
        return None

    # Unauthenticated: API gets JSON; browser gets redirected
    if _is_api_request(path):
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    # Preserve where the user was trying to go (external path if possible)
    next_external = request.headers.get("X-Original-URI") or request.full_path or path
    if next_external and next_external.endswith("?"):
        next_external = next_external[:-1]
    return redirect(f"{_login_path()}?next={quote(next_external or _external_path('/'))}")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = request.args.get("next") or request.form.get("next") or ""
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password and secrets.compare_digest(password, BEATS_PASSWORD):
            session[AUTH_SESSION_KEY] = True
            return redirect(_safe_next_url(unquote(next_url) if next_url else None))
        error = "Incorrect password."
    prefix = _external_prefix()
    login_action = f"{prefix}/login" if prefix else "/login"
    return render_template("login.html", error=error, login_action=login_action, next_url=next_url)


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.pop(AUTH_SESSION_KEY, None)
    return redirect(_login_path())

# Configuration
GENERATED_MUSIC_DIR = "/home/ubuntu/beats/generated_music"

# Trained model configuration
# Custom checkpoint mapping
CHECKPOINT_BASE_PATH = Path("/home/ubuntu/amanuel/beats/audiocraft_checkpoints/xps")
CHECKPOINT_MAPPING = {
    "original": "983190cf",      # Original dataset model
    "synthetic": "37f950a3",     # Synthetic dataset model
    "looperman": "b00102be",     # Looperman model (from scratch) - 1st place
    "looperman_from checkpoint": "e51bc38f"  # Looperman model (from 983190cf) - 2nd place
}

# Default checkpoint (for backward compatibility)
DEFAULT_CHECKPOINT = "original"
resTRAINED_CHECKPOINT_PATH = CHECKPOINT_BASE_PATH / CHECKPOINT_MAPPING[DEFAULT_CHECKPOINT] / "checkpoint.th"
TRAINED_BASE_MODEL_ID = "facebook/musicgen-stereo-melody-large"
TRAINED_LORA_RANK = 16
TRAINED_LORA_ALPHA = 32
DEFAULT_POST_PROCESSING_TRAINED = True
DEFAULT_POST_PROCESSING_BASIC = True
USE_CHECKPOINT_FOR_BASIC = True
APPLY_MASTERING_IN_POST = False

# Make sure we can import post-processing pipeline
POSTPROCESSING_DIR = "/home/ubuntu/beats/musicgen/postprocessing"
if os.path.exists(POSTPROCESSING_DIR) and POSTPROCESSING_DIR not in sys.path:
    sys.path.insert(0, POSTPROCESSING_DIR)

try:
    from musicgen_unified_postprocessor import MusicGenPostProcessor
    POSTPROCESSOR_AVAILABLE = True
except Exception as post_err:
    POSTPROCESSOR_AVAILABLE = False
    logger.warning(f"MusicGenPostProcessor unavailable: {post_err}")

try:
    from peft import LoraConfig
    from peft.tuners.lora import LoraModel
    PEFT_AVAILABLE = True
except Exception as peft_err:
    PEFT_AVAILABLE = False
    logger.warning(f"PEFT (LoRA) unavailable: {peft_err}")

# Ensure the generated music directory exists
os.makedirs(GENERATED_MUSIC_DIR, exist_ok=True)

# Store generation status for each request
generation_status = {}

class MusicGenerationTask:
    def __init__(self, task_id, prompt, duration=15, model_type="musicgen", checkpoint_id=DEFAULT_CHECKPOINT, 
                 lyrics="", inference_steps=48, guidance_scale=4.5,
                 post_process=False, target_key: Optional[str] = None, target_bpm: Optional[float] = None,
                 beat_name: Optional[str] = None):
        self.task_id = task_id
        self.prompt = prompt
        self.duration = duration
        self.model_type = model_type  # "musicgen" or "musicgen_trained"
        self.checkpoint_id = checkpoint_id  # Custom checkpoint ID: "fma", "original", or "synthetic"
        self.model_size = checkpoint_id  # Keep for backward compatibility
        self.lyrics = lyrics
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.status = "starting"
        self.progress = 0
        self.filename = None
        self.error = None
        self.start_time = time.time()
        self.post_process = post_process
        self.target_key = target_key
        self.target_bpm = target_bpm
        self.beat_name = beat_name  # User-provided beat name for filename


def normalize_key_value(key_str: Optional[str]) -> Optional[str]:
    """Normalize key strings into MusicGenPostProcessor format (e.g., F#-, C+)."""
    if not key_str:
        return None

    value = key_str.strip()
    if not value:
        return None

    value = value.replace("♭", "b").replace("♯", "#")
    value_lower = value.lower()

    quality = None
    if any(token in value_lower for token in ["minor", " min", " m", "-"]):
        quality = "-"
    elif any(token in value_lower for token in ["major", " maj", "+"]):
        quality = "+"

    # Replace words "sharp"/"flat"
    value = re.sub(r"sharp", "#", value, flags=re.IGNORECASE)
    value = re.sub(r"flat", "b", value, flags=re.IGNORECASE)

    match = re.search(r"([A-Ga-g])([#b]?)", value)
    if not match:
        return None

    note = match.group(1).upper()
    accidental = match.group(2)
    normalized = f"{note}{accidental}"

    if quality:
        normalized = f"{normalized}{quality}"

    return normalized


def infer_targets_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[float]]:
    """Infer target BPM and key from the prompt text."""
    if not prompt:
        return None, None

    bpm = None
    key = None

    bpm_match = re.search(r"(\d{2,3})\s*(?:bpm|beats per minute)", prompt, re.IGNORECASE)
    if bpm_match:
        try:
            bpm = float(bpm_match.group(1))
        except ValueError:
            bpm = None

    key_match = re.search(r"in\s+([A-G][#b]?)(?:\s*(major|minor|maj|min|\+|-|m))?", prompt, re.IGNORECASE)
    if key_match:
        key_candidate = key_match.group(1)
        quality = key_match.group(2) or ""
        key = normalize_key_value(f"{key_candidate}{quality}")
    else:
        # Look for shorthand like F#m or Gm
        shorthand_match = re.search(r"\b([A-G][#b]?)[-+]?m\b", prompt)
        if shorthand_match:
            key = normalize_key_value(f"{shorthand_match.group(1)} minor")

    return key, bpm


def apply_lora_to_model(model):
    """Attach LoRA adapters to the MusicGen transformer."""
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available; cannot apply LoRA adapters.")

    lora_config = LoraConfig(
        r=TRAINED_LORA_RANK,
        lora_alpha=TRAINED_LORA_ALPHA,
        target_modules=['out_proj', 'linear1', 'linear2'],
        lora_dropout=0.1,
        bias='none',
    )

    # LoraModel expects peft_config to be a dict mapping adapter_name -> config
    peft_config_dict = {'default': lora_config}
    model.lm.transformer = LoraModel(
        model.lm.transformer,
        peft_config_dict,
        'default'
    )

    return model

def create_modified_musicgen_script():
    """
    Create a modified version of musicgen_sample.py that accepts command line arguments
    and returns JSON output for better integration
    """
    script_content = '''#!/usr/bin/env python3
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
'''
    
    # Write the modified script
    with open("musicgen_api.py", "w") as f:
        f.write(script_content)
    
    logger.info("Created musicgen_api.py for Flask integration")

def generate_musicgen_trained(task):
    """
    Generate music using the trained MusicGen model
    """
    try:
        # Add audiocraft paths if not in venv
        import sys
        audiocraft_paths = [
            "/home/ubuntu/beats/musicgen/training/audiocraft",
            "/home/ubuntu/beats/musicgen/audiocraft"
        ]
        for path in audiocraft_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        # Import audiocraft first to ensure __version__ is set before importing models
        import audiocraft
        # Ensure __version__ exists (fallback if not set)
        if not hasattr(audiocraft, '__version__'):
            audiocraft.__version__ = '1.4.0a2'
        
        from audiocraft.models import MusicGen
        import torch
        import torchaudio
        
        task.status = "loading_model"
        task.progress = 10
        generation_status[task.task_id] = task
        
        logger.info("Loading trained MusicGen LoRA model...")
        
        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Aggressive memory clearing before loading
        def clear_gpu_memory():
            """Clear GPU memory aggressively"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
                gc.collect()
        torch.cuda.empty_cache()
        
        clear_gpu_memory()
        
        # Check available GPU memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Available GPU memory: {free_memory_gb:.2f} GB")
            if free_memory_gb < 2.0:
                logger.warning(f"Low GPU memory available: {free_memory_gb:.2f} GB. Generation may fail.")
        
        # Get checkpoint path based on task's checkpoint_id
        checkpoint_id = task.checkpoint_id if hasattr(task, 'checkpoint_id') else DEFAULT_CHECKPOINT
        if checkpoint_id not in CHECKPOINT_MAPPING:
            checkpoint_id = DEFAULT_CHECKPOINT
        checkpoint_path = CHECKPOINT_BASE_PATH / CHECKPOINT_MAPPING[checkpoint_id] / "checkpoint.th"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading base MusicGen model: {TRAINED_BASE_MODEL_ID}")
        model = MusicGen.get_pretrained(TRAINED_BASE_MODEL_ID, device='cuda')
        
        # Clear memory after model loading
        clear_gpu_memory()

        # Log compression model details
        logger.info(f"Compression model type: {type(model.compression_model).__name__}")
        logger.info(f"Compression model num_codebooks: {model.compression_model.num_codebooks}")
        logger.info(f"Compression model total_codebooks: {model.compression_model.total_codebooks}")
        logger.info(f"Compression model cardinality: {model.compression_model.cardinality}")
        logger.info(f"Compression model channels: {model.compression_model.channels}")
        logger.info(f"Compression model sample_rate: {model.compression_model.sample_rate}Hz")
        
        # CRITICAL: Try to increase EnCodec bandwidth for better quality (reduces noise)
        # Higher bandwidth = more codebooks = better fidelity
        # Training used n_q=4 (3kbps), but we can try to use more if available
        original_num_codebooks = model.compression_model.num_codebooks
        max_available_codebooks = model.compression_model.total_codebooks
        
        # Try to use 6-8 codebooks if available (6.0-12.0 kbps) for better quality
        # This significantly reduces quantization noise and improves beat clarity
        target_codebooks = min(8, max_available_codebooks)  # Use up to 8 codebooks
        
        if target_codebooks > original_num_codebooks and hasattr(model.compression_model, 'set_num_codebooks'):
            try:
                # Check if model supports higher codebooks
                if hasattr(model.compression_model, 'possible_num_codebooks'):
                    # HF EnCodec model - check available options
                    possible = model.compression_model.possible_num_codebooks
                    if target_codebooks in possible:
                        model.compression_model.set_num_codebooks(target_codebooks)
                        logger.info(f"✅ Increased EnCodec bandwidth: {original_num_codebooks} -> {target_codebooks} codebooks (better quality, less noise)")
                    else:
                        # Use highest available
                        max_possible = max([c for c in possible if c >= original_num_codebooks])
                        if max_possible > original_num_codebooks:
                            model.compression_model.set_num_codebooks(max_possible)
                            logger.info(f"✅ Increased EnCodec bandwidth: {original_num_codebooks} -> {max_possible} codebooks (best available)")
                        else:
                            logger.info(f"⚠️  EnCodec bandwidth limited to {original_num_codebooks} codebooks (training constraint)")
                else:
                    # Try to set directly (for standalone encodec)
                    model.compression_model.set_num_codebooks(target_codebooks)
                    logger.info(f"✅ Increased EnCodec bandwidth: {original_num_codebooks} -> {target_codebooks} codebooks")
            except Exception as e:
                logger.warning(f"Could not increase EnCodec bandwidth: {e}. Using {original_num_codebooks} codebooks.")
        
        # Log final codebook count
        final_num_codebooks = model.compression_model.num_codebooks
        if final_num_codebooks > original_num_codebooks:
            logger.info(f"📊 EnCodec bandwidth: {final_num_codebooks} codebooks (~{final_num_codebooks * 1.5:.1f} kbps) - optimized for quality")
        else:
            logger.info(f"📊 EnCodec bandwidth: {final_num_codebooks} codebooks (~{final_num_codebooks * 1.5:.1f} kbps) - training constraint")

        # CRITICAL FIX: Resize model embeddings AND output layers to match compression model cardinality
        # The checkpoint was trained with card=2048 but compression model has cardinality=1024
        # We need to resize both embedding layers and output projection layers
        if model.compression_model.cardinality < model.lm.card:
            logger.info(f"Resizing LM from card={model.lm.card} to card={model.compression_model.cardinality}...")
            model.lm.card = model.compression_model.cardinality

            # Resize all embedding layers (input)
            for i in range(model.lm.n_q):
                if hasattr(model.lm, 'emb'):
                    old_emb = model.lm.emb[i]
                    new_emb = torch.nn.Embedding(
                        model.compression_model.cardinality + 1,  # +1 for special token
                        old_emb.embedding_dim,
                        device=old_emb.weight.device,
                        dtype=old_emb.weight.dtype
                    )
                    with torch.no_grad():
                        copy_size = min(new_emb.weight.shape[0], old_emb.weight.shape[0])
                        new_emb.weight[:copy_size] = old_emb.weight[:copy_size]
                    model.lm.emb[i] = new_emb

            # Resize all output linear layers (output projections for each codebook)
            if hasattr(model.lm, 'linears'):
                for i in range(model.lm.n_q):
                    old_linear = model.lm.linears[i]
                    new_linear = torch.nn.Linear(
                        old_linear.in_features,
                        model.compression_model.cardinality,  # Output vocab size (no +1 here)
                        bias=old_linear.bias is not None,
                        device=old_linear.weight.device,
                        dtype=old_linear.weight.dtype
                    )
                    with torch.no_grad():
                        # Copy weights for the first cardinality tokens
                        copy_size = min(new_linear.weight.shape[0], old_linear.weight.shape[0])
                        new_linear.weight[:copy_size] = old_linear.weight[:copy_size]
                        if old_linear.bias is not None:
                            new_linear.bias[:copy_size] = old_linear.bias[:copy_size]
                    model.lm.linears[i] = new_linear

            logger.info(f"Resized {model.lm.n_q} embedding layers and {model.lm.n_q} output layers to card={model.lm.card}")

        logger.info("Applying LoRA adapters for inference...")
        model = apply_lora_to_model(model)

        logger.info(f"Loading trained weights from {checkpoint_path} (checkpoint: {checkpoint_id})")
        # Load checkpoint to CPU first to avoid OOM
        logger.info("Loading checkpoint to CPU first (to avoid OOM)...")
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

        state_dict = None
        if isinstance(checkpoint, dict):
            if 'best_state' in checkpoint and isinstance(checkpoint['best_state'], dict):
                state_dict = checkpoint['best_state'].get('model')
            if state_dict is None and 'state' in checkpoint and isinstance(checkpoint['state'], dict):
                state_dict = checkpoint['state'].get('model')
            if state_dict is None:
                state_dict = checkpoint.get('model')
        else:
            state_dict = checkpoint

        if state_dict is None:
            raise RuntimeError("Checkpoint state_dict missing 'model' weights.")

        # CRITICAL FIX: Handle cardinality mismatch
        # Checkpoint was trained with card=2048 but compression model has cardinality=1024
        # We need to truncate the embedding weights to match
        logger.info(f"Checkpoint card (from training): 2048")
        logger.info(f"Compression model cardinality: {model.compression_model.cardinality}")

        if model.compression_model.cardinality < 2048:
            logger.info("Truncating weights to match compression model cardinality...")
            truncated_keys = []
            emb_target_size = model.compression_model.cardinality + 1  # +1 for special token
            linear_target_size = model.compression_model.cardinality  # No +1 for output layers

            for key in list(state_dict.keys()):
                # Truncate embedding weights
                if key.startswith('emb.') and '.weight' in key:
                    if state_dict[key].shape[0] == 2049:  # 2048 + 1
                        original_shape = state_dict[key].shape
                        state_dict[key] = state_dict[key][:emb_target_size, :]
                        truncated_keys.append(key)
                        logger.info(f"  Truncated {key}: {original_shape} -> {state_dict[key].shape}")

                # Truncate output linear layer weights
                elif key.startswith('linears.') and '.weight' in key:
                    if state_dict[key].shape[0] == 2048:
                        original_shape = state_dict[key].shape
                        state_dict[key] = state_dict[key][:linear_target_size, :]
                        truncated_keys.append(key)
                        logger.info(f"  Truncated {key}: {original_shape} -> {state_dict[key].shape}")

                # Truncate output linear layer biases
                elif key.startswith('linears.') and '.bias' in key:
                    if state_dict[key].shape[0] == 2048:
                        original_shape = state_dict[key].shape
                        state_dict[key] = state_dict[key][:linear_target_size]
                        truncated_keys.append(key)
                        logger.info(f"  Truncated {key}: {original_shape} -> {state_dict[key].shape}")

            logger.info(f"Truncated {len(truncated_keys)} layers total")

        # Move state_dict to GPU incrementally to avoid OOM
        logger.info("Moving checkpoint weights to GPU (incremental to avoid OOM)...")
        state_dict_gpu = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict_gpu[key] = value.cuda()
            else:
                state_dict_gpu[key] = value
            # Clear CPU tensor immediately
            if isinstance(value, torch.Tensor):
                del value
        
        # Clear checkpoint from CPU memory
        del checkpoint
        del state_dict
        clear_gpu_memory()
        
        missing, unexpected = model.lm.load_state_dict(state_dict_gpu, strict=False)
        if missing or unexpected:
            logger.info(f"LoRA checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)} keys.")
        
        # Clear state_dict from GPU after loading
        del state_dict_gpu
        clear_gpu_memory()
        
        task.status = "generating"
        task.progress = 30
        generation_status[task.task_id] = task
        
        logger.info(f"Generating with trained MusicGen: prompt='{task.prompt}', duration={task.duration}")

        # Check model's max duration capability
        logger.info(f"Model max_duration: {model.max_duration}s, sample_rate: {model.sample_rate}Hz")
        logger.info(f"Requested duration: {task.duration}s")
        
        # Check available memory before generation
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Available GPU memory before generation: {free_memory_gb:.2f} GB")

        # Set generation parameters optimized for beat quality and noise reduction
        # These parameters are critical for reducing noise and improving beat clarity
        optimal_temperature = 0.75  # Lower = more stable, less noise (was 1.0)
        optimal_cfg = 5.5  # Higher = clearer beats, better guidance (was 3.0)
        optimal_top_p = 0.9  # Nucleus sampling for better quality (was 0.0)
        optimal_top_k = 250  # Keep top 250 tokens
        
        logger.info(f"🎛️  Generation parameters optimized for quality:")
        logger.info(f"   Temperature: {optimal_temperature} (lower = less noise, more stable beats)")
        logger.info(f"   CFG scale: {optimal_cfg} (higher = clearer beats, better guidance)")
        logger.info(f"   Top-p: {optimal_top_p} (nucleus sampling for quality)")
        logger.info(f"   Top-k: {optimal_top_k}")
        
        model.set_generation_params(
            duration=task.duration,
            temperature=optimal_temperature,
            cfg_coef=optimal_cfg,
            top_p=optimal_top_p,
            top_k=optimal_top_k,
            use_sampling=True,
            two_step_cfg=False
        )

        # Generate music (single pass, no chunking)
        descriptions = [task.prompt]
        logger.info(f"Generating audio with prompt: '{task.prompt}'")
        wav = model.generate(descriptions, progress=True)
        
        # Clear memory after generation
        clear_gpu_memory()

        actual_duration = wav.shape[-1] / model.sample_rate
        logger.info(f"Generated audio shape: {wav.shape}")
        logger.info(f"Actual generated duration: {actual_duration:.2f}s (requested: {task.duration}s)")

        # Remove trailing silence to ensure full duration contains actual music
        # This prevents silence padding from meeting the requested duration
        import torch
        import torchaudio
        
        # Convert to mono for silence detection if stereo
        if wav.dim() == 3:
            wav_mono = wav[0].mean(dim=0) if wav.shape[1] > 1 else wav[0, 0]
        else:
            wav_mono = wav.mean(dim=0) if wav.shape[0] > 1 else wav[0]
        
        # Find non-silent region (threshold: -40dB)
        silence_threshold = 10 ** (-40 / 20)  # -40dB in linear scale
        non_silent = torch.abs(wav_mono) > silence_threshold
        if non_silent.any():
            # Find first and last non-silent samples
            non_silent_indices = torch.where(non_silent)[0]
            first_non_silent = non_silent_indices[0].item()
            last_non_silent = non_silent_indices[-1].item()
            
            # Keep only the non-silent portion (with small padding to avoid cutting)
            padding_samples = int(0.1 * model.sample_rate)  # 100ms padding
            start_idx = max(0, first_non_silent - padding_samples)
            end_idx = min(wav.shape[-1], last_non_silent + padding_samples + 1)
            
            # Trim silence
            if wav.dim() == 3:
                wav = wav[:, :, start_idx:end_idx]
            else:
                wav = wav[:, start_idx:end_idx]
            
            trimmed_duration = wav.shape[-1] / model.sample_rate
            logger.info(f"Trimmed silence: {actual_duration:.2f}s -> {trimmed_duration:.2f}s (removed {actual_duration - trimmed_duration:.2f}s of silence)")
            
            # If trimmed audio is shorter than requested, we need to generate more
            # But don't pad with silence - instead log a warning
            if trimmed_duration < task.duration * 0.9:  # Allow 10% tolerance
                logger.warning(f"Generated audio ({trimmed_duration:.2f}s) is shorter than requested ({task.duration}s). Model may need longer generation time.")
        
        # Store original duration after trimming
        original_samples = wav.shape[-1]
        original_duration = original_samples / model.sample_rate

        processed_wav = wav
        # Apply post-processing with spectral denoising enabled for noise reduction
        # Even without key/BPM targets, we should apply denoising for MusicGen quantization noise
        if task.post_process and POSTPROCESSOR_AVAILABLE:
            try:
                logger.info(f"Applying post-processing: key={task.target_key}, bpm={task.target_bpm}, denoising=ENABLED")
                processor = MusicGenPostProcessor(sample_rate=model.sample_rate)
                processed_wav = processor.process(
                    audio=wav,
                    apply_mastering=APPLY_MASTERING_IN_POST,
                    target_key=task.target_key,
                    target_bpm=task.target_bpm,
                    return_tensor=True,
                    apply_denoise=True,  # ENABLE spectral denoising for MusicGen noise
                    apply_harmonic_exciter=True  # Enable high-end enhancement
                )

                # CRITICAL: Ensure post-processed audio has exact same duration
                if processed_wav.shape[-1] != original_samples:
                    import torch
                    current_samples = processed_wav.shape[-1]

                    if current_samples < original_samples:
                        # Pad with zeros to match original duration
                        pad_amount = original_samples - current_samples
                        processed_wav = torch.nn.functional.pad(processed_wav, (0, pad_amount))
                        logger.info(f"Padded {pad_amount} samples to preserve duration ({original_duration:.2f}s)")
                    else:
                        # Trim to match original duration
                        processed_wav = processed_wav[..., :original_samples]
                        logger.info(f"Trimmed to preserve duration ({original_duration:.2f}s)")

                if APPLY_MASTERING_IN_POST:
                    logger.info("Post-processing complete (mastering + key/BPM correction).")
                else:
                    logger.info("Post-processing complete (key/BPM correction only, mastering disabled).")
            except Exception as post_err:
                logger.warning(f"Post-processing failed, using raw audio: {post_err}")
                processed_wav = wav
        elif task.post_process and not POSTPROCESSOR_AVAILABLE:
            logger.warning("Post-processing requested but processor unavailable.")
        elif task.post_process and not (task.target_key or task.target_bpm):
            logger.info("Post-processing requested but no targets provided - skipping to avoid noise")

        # Audio output - upsample to 44.1kHz (CD quality) for high-quality output
        upsampled_wav = processed_wav
        final_sample_rate = model.sample_rate
        target_sample_rate = 44100  # CD quality (44.1kHz) - standard for music distribution
        
        if model.sample_rate == target_sample_rate:
            logger.info(f"✅ Audio is already at {final_sample_rate}Hz (native 44.1kHz quality, no resampling needed)")
        elif model.sample_rate != target_sample_rate:
            # Upsample to 44.1kHz using high-quality resampling
            try:
                logger.info(f"Upsampling audio from {model.sample_rate}Hz to {target_sample_rate}Hz (CD quality - 44.1kHz)...")
                import torchaudio
                
                # Use high-quality sinc interpolation with Kaiser window (best quality resampling)
                resampler = torchaudio.transforms.Resample(
                    model.sample_rate, 
                    target_sample_rate,
                    resampling_method='sinc_interp_kaiser'
                )
                upsampled_wav = resampler(processed_wav)
                final_sample_rate = target_sample_rate
                logger.info(f"✅ Audio upsampled to {final_sample_rate}Hz (CD quality) using high-quality sinc interpolation")
            except Exception as resample_err:
                logger.warning(f"Resampling failed: {resample_err}, using original audio at {model.sample_rate}Hz")
                upsampled_wav = processed_wav

        # Create filename - use beat_name if provided, otherwise use prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if task.beat_name:
            # Sanitize beat name for filename
            safe_name = "".join(c for c in task.beat_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')[:50]
            filename = f"{safe_name}_{timestamp}.wav"
        else:
            # Fallback to prompt-based filename
            safe_prompt = "".join(c for c in task.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]
            filename = f"musicgen_trained_{safe_prompt}_{timestamp}.wav"
        filepath = os.path.join(GENERATED_MUSIC_DIR, filename)
        
        # Save the audio file (use upsampled audio if available)
        sample_rate = final_sample_rate
        wav_to_save = upsampled_wav if upsampled_wav is not None else (processed_wav if processed_wav is not None else wav)
        wav_cpu = wav_to_save.detach().cpu()
        if wav_cpu.dim() == 3:
            wav_cpu = wav_cpu[0]
        torchaudio.save(filepath, wav_cpu, sample_rate)
        
        task.progress = 80
        generation_status[task.task_id] = task
        
        # Check if file was created successfully
        if os.path.exists(filepath):
            task.status = "success"
            task.progress = 100
            task.filename = filename
            generation_status[task.task_id] = task
            
            logger.info(f"Trained MusicGen generation complete: {filename}")
        else:
            raise Exception("Generated file not found")
        
        # Clean up memory
        try:
            del model
            clear_gpu_memory()
            logger.info("Trained MusicGen memory cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Memory cleanup warning: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"Error in trained MusicGen generation: {e}")
        # Ensure memory is cleared even on error
        try:
            if 'model' in locals():
                del model
            clear_gpu_memory()
        except:
            pass
        task.status = "error"
        task.error = f"Trained MusicGen generation failed: {str(e)}"
        task.progress = 0
        generation_status[task.task_id] = task
        

def run_musicgen_generation(task):
    """
    Run MusicGen generation in a separate thread
    """
    try:
        # Update status
        task.status = "initializing"
        generation_status[task.task_id] = task
        
        # Map medium to stereo-melody-large for pretrained models
        pretrained_model_size = "facebook/musicgen-stereo-melody-large" if task.model_size == "medium" else task.model_size
        
        # Prepare command - use venv Python if available
        venv_python = "/home/ubuntu/beats/venv/bin/python3"
        python_cmd = venv_python if os.path.exists(venv_python) else "python3"
        
        cmd = [
            python_cmd, "musicgen_api.py",
            "--prompt", task.prompt,
            "--duration", str(task.duration),
            "--model_size", pretrained_model_size,
            "--output_dir", GENERATED_MUSIC_DIR
        ]
        
        logger.info(f"Starting MusicGen with command: {' '.join(cmd)}")
        
        # Run the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in process.stdout:
            if line.strip():
                try:
                    status_update = json.loads(line.strip())
                    task.status = status_update.get("status", "running")
                    
                    if task.status == "success":
                        task.filename = status_update.get("filename")
                        task.progress = 100
                    elif task.status == "error":
                        task.error = status_update.get("message", "Unknown error")
                        task.progress = 0
                    else:
                        # Update progress based on status
                        if task.status == "loading_model":
                            task.progress = 20
                        elif task.status == "generating":
                            task.progress = 50
                        
                    generation_status[task.task_id] = task
                    logger.info(f"Task {task.task_id}: {task.status} - {task.progress}%")
                    
                except json.JSONDecodeError:
                    # Handle non-JSON output (like progress bars)
                    logger.info(f"Non-JSON output: {line.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            # Handle error
            stderr_output = process.stderr.read()
            task.status = "error"
            task.error = f"Process failed with return code {return_code}: {stderr_output}"
            task.progress = 0
        elif task.status != "success":
            # If we didn't get a success status, assume it completed
            task.status = "completed"
            task.progress = 100
            
    except Exception as e:
        logger.error(f"Error in MusicGen generation: {e}")
        task.status = "error"
        task.error = str(e)
        task.progress = 0
    
    finally:
        generation_status[task.task_id] = task

def run_music_generation(task):
    """
    Route to appropriate generation method based on model type
    """
    # Determine pipeline usage:
    # - Basic mode: always use trained (USE_CHECKPOINT_FOR_BASIC)
    # - Advanced mode: use trained only for "large", pretrained for "medium"/"small"
    # - Explicit musicgen_trained: always use trained
    use_trained_pipeline = (
        task.model_type == "musicgen_trained" or
        (task.model_type == "musicgen" and USE_CHECKPOINT_FOR_BASIC) or  # Basic mode
        (task.model_type == "musicgen" and task.model_size == "large")  # Advanced mode large
    )

    if use_trained_pipeline:
        # Use trained MusicGen model/checkpoint path
        generate_musicgen_trained(task)
    else:  # musicgen (pretrained)
        run_musicgen_generation(task)

@app.route('/')
def index():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_music():
    """Start music generation process"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or not data.get('prompt'):
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400
        
        # Extract parameters
        prompt = data.get('prompt').strip()
        default_duration = int(data.get('duration', 15))
        model_type = data.get('modelType', 'musicgen').lower()  # 'musicgen' or 'musicgen_trained'
        # Support both old model_size and new checkpoint_id
        checkpoint_id = data.get('checkpointId', data.get('modelSize', DEFAULT_CHECKPOINT))
        model_size = data.get('modelSize', 'medium')  # For backward compatibility
        lyrics = data.get('lyrics', '')
        inference_steps = int(data.get('inferenceSteps', 48))
        guidance_scale = float(data.get('guidanceScale', 4.5))
        beat_name = data.get('beatName', '').strip()  # User-provided beat name for filename

        # Always use the duration from the dedicated field, never parse from prompt
        duration = default_duration
        
        # Validate parameters
        if duration < 5 or duration > 300:
            return jsonify({'success': False, 'error': 'Duration must be between 5 and 300 seconds (5 minutes)'}), 400
        
        if model_type not in ['musicgen', 'musicgen_trained']:
            return jsonify({'success': False, 'error': 'Invalid model type. Use "musicgen" or "musicgen_trained"'}), 400
        
        # Validate checkpoint_id
        if checkpoint_id not in CHECKPOINT_MAPPING:
            # Fallback: if old model_size format, map to default checkpoint
            if checkpoint_id in ['small', 'medium', 'large']:
                checkpoint_id = DEFAULT_CHECKPOINT
            else:
                return jsonify({'success': False, 'error': f'Invalid checkpoint. Use one of: {", ".join(CHECKPOINT_MAPPING.keys())}'}), 400
        
        # Determine pipeline usage:
        # - Basic mode: always use trained (USE_CHECKPOINT_FOR_BASIC)
        # - Advanced mode: use trained checkpoint (custom checkpoints)
        # - Explicit musicgen_trained: always use trained
        use_trained_pipeline = (
            model_type == 'musicgen_trained' or
            (model_type == 'musicgen' and USE_CHECKPOINT_FOR_BASIC) or  # Basic mode
            (model_type == 'musicgen' and checkpoint_id in CHECKPOINT_MAPPING)  # Custom checkpoint
        )

        # Post-processing controls
        user_post_process = data.get('postProcess')
        if isinstance(user_post_process, str):
            user_post_process = user_post_process.strip().lower() in ("1", "true", "yes", "on")
        default_post_process = DEFAULT_POST_PROCESSING_TRAINED if use_trained_pipeline else DEFAULT_POST_PROCESSING_BASIC
        post_process = bool(user_post_process) if user_post_process is not None else default_post_process

        raw_target_key = data.get('targetKey') or data.get('target_key')
        target_key = normalize_key_value(raw_target_key) if raw_target_key else None

        raw_target_bpm = data.get('targetBpm') or data.get('target_bpm')
        target_bpm = None
        if raw_target_bpm is not None:
            try:
                target_bpm = float(str(raw_target_bpm).strip())
            except (TypeError, ValueError):
                logger.warning(f"Invalid target BPM '{raw_target_bpm}', ignoring.")

        inferred_key, inferred_bpm = infer_targets_from_prompt(prompt)
        if target_key is None:
            target_key = inferred_key
        if target_bpm is None:
            target_bpm = inferred_bpm

        # Only enable post-processing if we have actual targets to correct
        # Without targets, post-processing would just add noise without benefit
        if post_process and not target_key and not target_bpm:
            logger.info("Post-processing requested but no targets provided - disabling to avoid noise")
            post_process = False

        if post_process:
            logger.info(f"Post-processing enabled (target_key={target_key}, target_bpm={target_bpm})")
        else:
            logger.info("Post-processing disabled - using raw model output for best quality")
        
        # Create task
        task_id = str(uuid.uuid4())
        task = MusicGenerationTask(
            task_id, prompt, duration, model_type, checkpoint_id,
            lyrics, inference_steps, guidance_scale,
            post_process=post_process,
            target_key=target_key,
            target_bpm=target_bpm,
            beat_name=beat_name
        )
        
        # Start generation in background thread
        thread = threading.Thread(target=run_music_generation, args=(task,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started {model_type} generation task {task_id} for prompt: '{prompt}'")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'model_type': model_type,
            'message': f'{model_type.upper()} generation started'
        })
        
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Get the status of a music generation task"""
    if task_id not in generation_status:
        return jsonify({'success': False, 'error': 'Task not found'}), 404
    
    task = generation_status[task_id]
    
    response = {
        'task_id': task_id,
        'status': task.status,
        'progress': task.progress,
        'model_type': task.model_type,
        'elapsed_time': time.time() - task.start_time
    }
    
    if task.status == 'success' and task.filename:
        response['filename'] = task.filename
        response['download_url'] = f'/api/download/{task.filename}'
    
    if task.status == 'error' and task.error:
        response['error'] = task.error
    
    return jsonify(response)

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated music file"""
    filepath = os.path.join(GENERATED_MUSIC_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Send file with proper MIME type for audio
    return send_file(filepath, as_attachment=False, mimetype='audio/wav')

# Section-based composition storage
section_compositions = {}  # task_id -> composition data

class SectionDefinition:
    """Represents a single section of a composition"""
    def __init__(self, name: str, bars: int = 4, chord_progression: str = "", 
                 instrumentation: List[str] = None, style: str = "", 
                 tempo: Optional[float] = None, key: Optional[str] = None,
                 pitch_curve: str = "stable"):
        self.name = name
        self.bars = bars
        self.chord_progression = chord_progression
        self.instrumentation = instrumentation or []
        self.style = style
        self.tempo = tempo
        self.key = key
        self.pitch_curve = pitch_curve
        self.audio_data = None
        self.audio_file = None
        self.duration = None  # Will be calculated from bars and tempo

def calculate_section_duration(bars: int, tempo: float = 120.0) -> float:
    """Calculate duration in seconds from bars and tempo"""
    # 4/4 time signature assumed
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration_seconds = (total_beats / tempo) * 60.0
    return duration_seconds

def build_section_prompt(section: SectionDefinition, base_style: str = "") -> str:
    """Build a MusicGen prompt from section definition"""
    parts = []
    
    # Section name and style
    if section.style:
        parts.append(section.style)
    elif base_style:
        parts.append(base_style)
    
    # Section type
    section_descriptors = {
        "Intro": "intro section, building up gradually",
        "Middle": "middle section, main theme",
        "Bridge": "bridge section, transition",
        "Drop": "drop section, high energy climax",
        "Chorus": "chorus section, memorable hook",
        "Ending": "ending section, fade out"
    }
    if section.name in section_descriptors:
        parts.append(section_descriptors[section.name])
    
    # Chord progression
    if section.chord_progression:
        parts.append(f"chord progression: {section.chord_progression}")
    
    # Instrumentation
    if section.instrumentation:
        instruments = ", ".join(section.instrumentation)
        parts.append(f"instruments: {instruments}")
    
    # Tempo
    if section.tempo:
        parts.append(f"{int(section.tempo)} bpm")
    
    # Key
    if section.key:
        parts.append(f"in {section.key}")
    
    # Pitch curve
    if section.pitch_curve and section.pitch_curve != "stable":
        parts.append(f"pitch: {section.pitch_curve}")
    
    return ", ".join(parts) if parts else f"{section.name} section"

def combine_audio_sections(sections: List[Dict[str, Any]], crossfade_duration: float = 1.5) -> Tuple[Any, int]:
    """
    Combine multiple audio sections with smooth transitions using improved crossfading
    
    Args:
        sections: List of section dicts with 'audio' (tensor) and 'sample_rate'
        crossfade_duration: Duration of crossfade between sections in seconds (default 1.5s for smoother transitions)
    
    Returns:
        Combined audio tensor and sample rate
    """
    import torch
    import torchaudio
    import math
    
    if not sections:
        raise ValueError("No sections to combine")
    
    combined_audio = None
    sample_rate = sections[0]['sample_rate']
    
    for i, section in enumerate(sections):
        audio = section['audio']
        sr = section['sample_rate']
        
        logger.info(f"Processing section {i+1}: shape={audio.shape}, sample_rate={sr}")
        
        # Normalize audio shape: torchaudio.load returns [channels, samples] or [batch, channels, samples]
        # We want [channels, samples]
        if audio.dim() == 3:
            # [batch, channels, samples] -> [channels, samples]
            audio = audio[0]
        elif audio.dim() == 1:
            # [samples] -> [1, samples] (mono)
            audio = audio.unsqueeze(0)
        # audio.dim() == 2 means [channels, samples] which is what we want
        
        # Resample if needed
        if sr != sample_rate:
            logger.info(f"Resampling section {i+1} from {sr}Hz to {sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(sr, sample_rate, resampling_method='sinc_interp_kaiser')
            audio = resampler(audio)
        
        # Ensure stereo: [channels, samples] format
        if audio.shape[0] == 1:
            # Mono to stereo: duplicate channel
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            # More than 2 channels: take first 2
            audio = audio[:2, :]
        
        # Validate audio has content
        if audio.numel() == 0:
            raise ValueError(f"Section {i+1} has no audio data")
        
        # Normalize volume of each section to prevent sudden volume changes
        max_val = torch.abs(audio).max()
        if max_val > 0:
            # Normalize to 0.8 to leave headroom for mixing
            audio = audio * (0.8 / max_val)
        
        logger.info(f"Section {i+1} processed: shape={audio.shape}, samples={audio.shape[1]}")
        
        # Apply smooth crossfade with pre-fade technique
        # Next section starts playing at low volume before previous section ends
        if combined_audio is not None and crossfade_duration > 0:
            crossfade_samples = int(crossfade_duration * sample_rate)
            
            # Calculate crossfade region (use up to 15% of each section for smoother transitions)
            max_crossfade = min(
                int(combined_audio.shape[-1] * 0.15),  # 15% of previous section
                int(audio.shape[-1] * 0.15),            # 15% of new section
                crossfade_samples
            )
            
            # Ensure minimum crossfade of 1 second for smoothness
            min_crossfade = int(1.0 * sample_rate)
            max_crossfade = max(max_crossfade, min_crossfade)
            
            # Don't exceed available samples
            max_crossfade = min(
                combined_audio.shape[-1] // 2,
                audio.shape[-1] // 2,
                max_crossfade
            )
            
            if max_crossfade > 0:
                logger.info(f"Applying {max_crossfade} sample crossfade ({max_crossfade/sample_rate:.2f}s) with pre-fade technique")
                
                import math
                
                # Create fade curves for smooth transition
                x = torch.linspace(0, math.pi, max_crossfade, device=audio.device, dtype=audio.dtype)
                
                # Previous section: fade out from 1.0 to 0.0 (cosine curve)
                fade_out = (torch.cos(x) + 1.0) / 2.0  # Maps [0, pi] to [1, 0]
                
                # New section: fade in from 0.25 to 1.0 (starts at 25% volume, rises to 100%)
                # This creates the "background playing" effect
                fade_in_base = 0.25  # Start at 25% volume
                fade_in_range = 0.75  # Rise to 100% (75% increase)
                fade_in = fade_in_base + (fade_in_range * (torch.cos(x - math.pi) + 1.0) / 2.0)
                
                # Apply fade out to end of previous section
                combined_audio[:, -max_crossfade:] *= fade_out.unsqueeze(0)
                
                # Apply fade in to start of new section (starts quiet, gets louder)
                audio[:, :max_crossfade] *= fade_in.unsqueeze(0)
                
                # Overlap and mix (additive mixing)
                # The new section starts playing quietly in the background
                combined_audio[:, -max_crossfade:] += audio[:, :max_crossfade]
                
                # Remove the crossfaded portion from new section
                audio = audio[:, max_crossfade:]
        
        # Concatenate
        if combined_audio is None:
            combined_audio = audio
        else:
            combined_audio = torch.cat([combined_audio, audio], dim=-1)
    
    # Final normalization to prevent clipping after mixing
    max_val = torch.abs(combined_audio).max()
    if max_val > 1.0:
        logger.info(f"Normalizing combined audio (max was {max_val:.3f})")
        combined_audio = combined_audio / max_val * 0.95  # Leave 5% headroom
    
    # Soft clipping to prevent harsh artifacts
    combined_audio = torch.tanh(combined_audio * 0.9) * 0.95
    
    logger.info(f"Combined audio final shape: {combined_audio.shape}, total samples: {combined_audio.shape[1]}, duration: {combined_audio.shape[1]/sample_rate:.2f}s")
    return combined_audio, sample_rate

@app.route('/api/generate-sections', methods=['POST'])
def generate_sections():
    """Generate music composition with multiple sections"""
    try:
        data = request.get_json()
        
        if not data or 'sections' not in data:
            return jsonify({'success': False, 'error': 'Sections are required'}), 400
        
        sections_data = data.get('sections', [])
        if not sections_data:
            return jsonify({'success': False, 'error': 'At least one section is required'}), 400
        
        # Validate sections
        valid_sections = ["Intro", "Middle", "Bridge", "Drop", "Chorus", "Ending"]
        for section in sections_data:
            if section.get('name') not in valid_sections:
                return jsonify({
                    'success': False, 
                    'error': f"Invalid section name: {section.get('name')}. Valid: {', '.join(valid_sections)}"
                }), 400
        
        # Extract global parameters
        global_tempo = data.get('tempo', 120.0)
        global_key = data.get('key', 'C major')
        model_type = data.get('modelType', 'music-chain').lower()
        # Map 'music-chain' to use trained checkpoint (same as 'musicgen_trained')
        if model_type == 'music-chain':
            model_type = 'musicgen_trained'
        # Support both checkpointId and modelSize for backward compatibility
        checkpoint_id = data.get('checkpointId', data.get('modelSize', DEFAULT_CHECKPOINT))
        if checkpoint_id not in CHECKPOINT_MAPPING:
            checkpoint_id = DEFAULT_CHECKPOINT
        model_size = checkpoint_id  # Keep for backward compatibility
        base_style = data.get('base_style', '')
        # Do NOT combine automatically - only when explicitly requested
        combine_track = False
        
        # Create task
        task_id = str(uuid.uuid4())
        
        # Parse sections
        sections = []
        for section_data in sections_data:
            bars = section_data.get('bars', 4)
            tempo = section_data.get('tempo', global_tempo)
            duration = calculate_section_duration(bars, tempo)
            
            section = SectionDefinition(
                name=section_data.get('name', 'Middle'),
                bars=bars,
                chord_progression=section_data.get('chord_progression', ''),
                instrumentation=section_data.get('instrumentation', []),
                style=section_data.get('style', ''),
                tempo=tempo,
                # Use global key for all sections; per-section key removed
                key=global_key,
                pitch_curve=section_data.get('pitch_curve', 'stable')
            )
            section.duration = duration
            sections.append(section)
        
        # Store composition
        composition = {
            'task_id': task_id,
            'sections': sections,
            'global_tempo': global_tempo,
            'global_key': global_key,
            'model_type': model_type,
            'checkpoint_id': checkpoint_id,
            'model_size': model_size,  # Keep for backward compatibility
            'base_style': base_style,
            'combine_track': combine_track,
            'status': 'generating',
            'generated_sections': [],
            'start_time': time.time()
        }
        section_compositions[task_id] = composition
        
        # Start generation in background
        thread = threading.Thread(target=generate_sections_background, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started section-based generation task {task_id} with {len(sections)} sections")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': f'Section-based generation started with {len(sections)} sections'
        })
        
    except Exception as e:
        logger.error(f"Error starting section generation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_sections_background(task_id: str):
    """Background thread to generate all sections individually"""
    try:
        composition = section_compositions.get(task_id)
        if not composition:
            logger.error(f"Composition {task_id} not found")
            return
        
        sections = composition['sections']
        model_type = composition['model_type']
        checkpoint_id = composition.get('checkpoint_id', composition.get('model_size', DEFAULT_CHECKPOINT))
        global_tempo = composition['global_tempo']
        global_key = composition['global_key']
        
        generated_sections = []
        
        for i, section in enumerate(sections):
            logger.info(f"Generating section {i+1}/{len(sections)}: {section.name}")
            
            # Build prompt
            base_style = composition.get('base_style', '')
            prompt = build_section_prompt(section, base_style)
            
            # Create a temporary task for this section
            section_task = MusicGenerationTask(
                task_id=f"{task_id}_section_{i}",
                prompt=prompt,
                duration=section.duration,
                model_type=model_type,
                checkpoint_id=checkpoint_id,
                target_bpm=section.tempo,
                target_key=section.key
            )
            
            # Generate section
            if model_type == 'musicgen_trained':
                generate_musicgen_trained(section_task)
            else:
                run_musicgen_generation(section_task)
            
            if section_task.status == 'success' and section_task.filename:
                # Load generated audio
                filepath = os.path.join(GENERATED_MUSIC_DIR, section_task.filename)
                import torchaudio
                wav, sr = torchaudio.load(filepath)
                
                section_result = {
                    'name': section.name,
                    'bars': section.bars,
                    'chord_progression': section.chord_progression,
                    'instrumentation': section.instrumentation,
                    'style': section.style,
                    'audio_parameters': {
                        'tempo': section.tempo,
                        'key': section.key,
                        'pitch_curve': section.pitch_curve
                    },
                    'audio': wav,
                    'sample_rate': sr,
                    'duration': section.duration,
                    'filename': section_task.filename
                }
                generated_sections.append(section_result)
                
                # Update composition with this section immediately (so UI can access it)
                composition['generated_sections'] = generated_sections
                
                logger.info(f"Section {section.name} generated successfully - available at /api/download/{section_task.filename}")
            else:
                raise Exception(f"Section {section.name} generation failed: {section_task.error}")
        
        # All sections generated - mark as complete (but NOT combined)
        composition['status'] = 'success'
        logger.info(f"All {len(generated_sections)} sections generated for task {task_id}. Ready for combination if requested.")
        
    except Exception as e:
        logger.error(f"Error in section generation background: {e}")
        if task_id in section_compositions:
            section_compositions[task_id]['status'] = 'error'
            section_compositions[task_id]['error'] = str(e)

@app.route('/api/sections-status/<task_id>')
def get_sections_status(task_id):
    """Get status of section-based generation - returns sections as they're generated"""
    if task_id not in section_compositions:
        return jsonify({'success': False, 'error': 'Task not found'}), 404
    
    composition = section_compositions[task_id]
    
    response = {
        'task_id': task_id,
        'status': composition['status'],
        'total_sections': len(composition['sections']),
        'generated_sections': len(composition.get('generated_sections', [])),
        'elapsed_time': time.time() - composition['start_time']
    }
    
    # Return section metadata with audio URLs (available as they're generated)
    # This works for both 'generating' and 'success' status
    sections_metadata = []
    for section in composition.get('generated_sections', []):
        sections_metadata.append({
            'name': section['name'],
            'bars': section['bars'],
            'chord_progression': section['chord_progression'],
            'instrumentation': section['instrumentation'],
            'style': section['style'],
            'audio_parameters': section['audio_parameters'],
            'duration': section['duration'],
            'filename': section['filename'],
            'download_url': f"/beats/api/download/{section['filename']}",
            'audio_url': f"/beats/api/download/{section['filename']}"  # For UI audio player
        })
    
    response['sections'] = sections_metadata
    
    # Include combined track if it exists (only after explicit combine request)
    if 'combined_track' in composition:
        response['combined_track'] = {
            'filename': composition['combined_track']['filename'],
            'download_url': f"/beats/api/download/{composition['combined_track']['filename']}",
            'audio_url': f"/beats/api/download/{composition['combined_track']['filename']}"
        }
    
    if composition['status'] == 'error':
        response['error'] = composition.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/combine-sections/<task_id>', methods=['POST'])
def combine_sections(task_id):
    """Combine previously generated sections into a full track"""
    logger.info(f"Combine request for task_id: {task_id}")
    logger.info(f"Available task IDs: {list(section_compositions.keys())}")
    
    if task_id not in section_compositions:
        logger.error(f"Task {task_id} not found in section_compositions. Available: {list(section_compositions.keys())}")
        return jsonify({'success': False, 'error': f'Task not found: {task_id}'}), 404
    
    composition = section_compositions[task_id]
    
    if composition['status'] != 'success':
        return jsonify({'success': False, 'error': 'All sections must be generated before combining'}), 400
    
    generated_sections = composition.get('generated_sections', [])
    if len(generated_sections) < 2:
        return jsonify({'success': False, 'error': 'At least 2 sections are required for combination'}), 400
    
    try:
        # Get custom order from request if provided
        data = request.get_json() or {}
        section_order = data.get('section_order')
        
        # If custom order provided, reorder sections accordingly
        if section_order and len(section_order) == len(generated_sections):
            # Create a mapping of filename to section
            section_map = {s['filename']: s for s in generated_sections}
            # Reorder sections based on provided order
            ordered_sections = []
            for filename in section_order:
                if filename in section_map:
                    ordered_sections.append(section_map[filename])
            if len(ordered_sections) == len(generated_sections):
                generated_sections = ordered_sections
                logger.info(f"Using custom section order: {section_order}")
            else:
                logger.warning(f"Custom order incomplete, using original order")
        else:
            logger.info(f"Using original section generation order")
        
        logger.info(f"Combining {len(generated_sections)} sections for task {task_id}...")
        
        # Load audio from files (reuse previously generated audio)
        import torchaudio
        sections_to_combine = []
        for section in generated_sections:
            filepath = os.path.join(GENERATED_MUSIC_DIR, section['filename'])
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Section audio file not found: {filepath}")
            
            wav, sr = torchaudio.load(filepath)
            sections_to_combine.append({
                'audio': wav,
                'sample_rate': sr
            })
        
        # Combine sections
        combined_audio, sample_rate = combine_audio_sections(sections_to_combine)
        
        # Save combined track
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"composition_{task_id}_{timestamp}.wav"
        combined_filepath = os.path.join(GENERATED_MUSIC_DIR, combined_filename)
        
        logger.info(f"Saving combined track: shape={combined_audio.shape}, sample_rate={sample_rate}")
        
        # torchaudio.save expects [channels, samples]
        audio_to_save = combined_audio
        if combined_audio.dim() == 1:
            audio_to_save = combined_audio.unsqueeze(0)
        elif combined_audio.dim() > 2:
            if combined_audio.shape[0] == 1:
                audio_to_save = combined_audio.squeeze(0)
            else:
                raise ValueError(f"Unexpected combined audio shape: {combined_audio.shape}")
        
        torchaudio.save(combined_filepath, audio_to_save, sample_rate)
        
        # Verify file was created
        if not os.path.exists(combined_filepath):
            raise Exception(f"Combined track file was not created: {combined_filepath}")
        
        file_size = os.path.getsize(combined_filepath)
        logger.info(f"Combined track saved: {combined_filename} ({file_size} bytes)")
        
        if file_size < 1000:
            logger.warning(f"Combined track file is very small ({file_size} bytes), may be corrupted")
        
        # Store combined track info
        composition['combined_track'] = {
            'filename': combined_filename,
            'sample_rate': sample_rate
        }
        
        return jsonify({
            'success': True,
            'filename': combined_filename,
            'download_url': f"/beats/api/download/{combined_filename}",
            'audio_url': f"/beats/api/download/{combined_filename}",
            'message': f'Successfully combined {len(generated_sections)} sections'
        })
        
    except Exception as e:
        logger.error(f"Error combining sections: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files')
def list_files():
    """List all generated music files"""
    try:
        files = []
        if os.path.exists(GENERATED_MUSIC_DIR):
            for filename in os.listdir(GENERATED_MUSIC_DIR):
                if filename.endswith(('.wav', '.mp3')):
                    filepath = os.path.join(GENERATED_MUSIC_DIR, filename)
                    stat = os.stat(filepath)
                    
                    # Detect model type from filename
                    model_type = 'musicgen_trained' if filename.startswith('musicgen_trained_') else 'musicgen'
                    
                    files.append({
                        'filename': filename,
                        'model_type': model_type,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'download_url': f'/api/download/{filename}'
                    })
        
        files.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({'files': files})
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def list_models():
    """List available models"""
    return jsonify({
        'models': [
            {
                'id': 'musicgen',
                'name': 'MusicGen',
                'description': 'Meta\'s MusicGen model (local)',
                'sizes': ['small', 'medium', 'large'],
                'supports_lyrics': False
            },
            {
                'id': 'musicgen_trained',
                'name': 'MusicGen (Trained)',
                'description': 'Trained MusicGen model (local)',
                'sizes': ['small', 'medium', 'large'],
                'supports_lyrics': False
            }
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def _register_prefixed_routes():
    """
    Register /beats-prefixed aliases for all user-facing routes.

    The frontend JS calls endpoints under /beats (e.g. /beats/api/generate).
    In some deployments an upstream proxy strips /beats before forwarding to Flask,
    but when accessing Flask directly (e.g. from another device), these aliases
    are required.
    """
    if not URL_PREFIX:
        return

    # Main UI
    app.add_url_rule(f"{URL_PREFIX}", endpoint="beats_index", view_func=index, methods=["GET"])
    app.add_url_rule(f"{URL_PREFIX}/", endpoint="beats_index_slash", view_func=index, methods=["GET"])

    # Auth aliases
    app.add_url_rule(f"{URL_PREFIX}/login", endpoint="beats_login", view_func=login, methods=["GET", "POST"])
    app.add_url_rule(f"{URL_PREFIX}/logout", endpoint="beats_logout", view_func=logout, methods=["GET", "POST"])

    # API aliases
    app.add_url_rule(f"{URL_PREFIX}/api/generate", endpoint="beats_generate_music", view_func=generate_music, methods=["POST"])
    app.add_url_rule(f"{URL_PREFIX}/api/status/<task_id>", endpoint="beats_get_status", view_func=get_status, methods=["GET"])
    app.add_url_rule(f"{URL_PREFIX}/api/download/<filename>", endpoint="beats_download_file", view_func=download_file, methods=["GET"])
    app.add_url_rule(f"{URL_PREFIX}/api/generate-sections", endpoint="beats_generate_sections", view_func=generate_sections, methods=["POST"])
    app.add_url_rule(f"{URL_PREFIX}/api/sections-status/<task_id>", endpoint="beats_get_sections_status", view_func=get_sections_status, methods=["GET"])
    app.add_url_rule(f"{URL_PREFIX}/api/combine-sections/<task_id>", endpoint="beats_combine_sections", view_func=combine_sections, methods=["POST"])
    app.add_url_rule(f"{URL_PREFIX}/api/files", endpoint="beats_list_files", view_func=list_files, methods=["GET"])
    app.add_url_rule(f"{URL_PREFIX}/api/models", endpoint="beats_list_models", view_func=list_models, methods=["GET"])

if __name__ == '__main__':
    # Create the modified musicgen script on startup
    create_modified_musicgen_script()
    _register_prefixed_routes()
    
    logger.info("Starting Flask server...")
    logger.info(f"Generated music will be saved to: {os.path.abspath(GENERATED_MUSIC_DIR)}")
    logger.info("Supported models: MusicGen (pretrained), MusicGen (trained)")
    logger.info(f"App will be available at: http://<host>:5000{URL_PREFIX or ''}")
    
    # Run the Flask app
    # For production, set debug=False and use a production WSGI server
    app.run(
        host=os.environ.get("FLASK_HOST", "0.0.0.0"),
        port=5000,
        debug=False,  # Disable debug mode for production
        threaded=True
    )
