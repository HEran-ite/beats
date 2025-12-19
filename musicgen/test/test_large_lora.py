#!/usr/bin/env python3
"""
Test Pretrained LARGE vs Trained LARGE LoRA MusicGen with Real Test Dataset
Uses 20 prompts from actual test data with ground truth BPM/key/style
Generates audio for both models, analyzes them, and compares results
"""
import sys
sys.path.insert(0, '/home/ubuntu/beats/musicgen/training/audiocraft')

import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import gc
import json
import random

# Audio analysis
import essentia.standard as es

from audiocraft.models import MusicGen
from peft import LoraConfig
from peft.tuners.lora import LoraModel

# Post-processing
sys.path.insert(0, '/home/ubuntu/beats/musicgen/postprocessing')
from musicgen_unified_postprocessor import MusicGenPostProcessor

# Paths
PRETRAINED_MODEL = "facebook/musicgen-large"
TRAINED_CHECKPOINT = Path("/home/ubuntu/amanuel/beats/audiocraft_checkpoints/xps/b00102be")
TEST_DATASET = "/home/ubuntu/beats/musicgen/training/audiocraft/egs/looperman_audio/evaluate"
OUTPUT_DIR = Path("/home/ubuntu/beats/musicgen/performance/looperman")
LOG_FILE = OUTPUT_DIR / "test_large_lora_looperman.log"

def keys_match(key1: str, key2: str) -> bool:
    """
    Check if two keys match, considering enharmonic equivalents (mode-agnostic).
    Only compares pitch class - ignores major/minor mode since Essentia's
    mode detection is unreliable.
    """
    k1 = key1.lower().strip()
    k2 = key2.lower().strip()

    if k1 == k2:
        return True

    enharmonic = {
        'c#': 'db', 'db': 'c#',
        'd#': 'eb', 'eb': 'd#',
        'f#': 'gb', 'gb': 'f#',
        'g#': 'ab', 'ab': 'g#',
        'a#': 'bb', 'bb': 'a#',
    }

    def parse_key(k):
        parts = k.split()
        if len(parts) == 2:
            return parts[0], parts[1]
        return k, ''

    note1, mode1 = parse_key(k1)
    note2, mode2 = parse_key(k2)

    if note1 == note2:
        return True

    if note1 in enharmonic and enharmonic[note1] == note2:
        return True

    return False

def load_test_cases(dataset_path, num_samples=20, start_from=0):
    """Load test cases from dataset (JSONL format) - supports beats_full_complete and looperman_audio"""
    print(f"\n📂 Loading test cases from: {dataset_path}")

    metadata_file = Path(dataset_path) / "data.jsonl"

    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return []

    evaluate_tracks = []
    with open(metadata_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                meta = data.get('meta', {})
                description = data.get('description', '')
                
                # Parse key format: 
                # - beats_full_complete: "D" = "D major", "E-" = "E minor"
                # - looperman_audio: "D" = "D major", "Dm" = "D minor", "D#" = "D# major"
                key_raw = meta.get('key', '')
                if not key_raw:
                    key_full = "C major"
                elif key_raw.endswith('-'):
                    key_note = key_raw[:-1]
                    key_full = f"{key_note} minor"
                elif key_raw.lower().endswith('m'):
                    key_note = key_raw[:-1]
                    key_full = f"{key_note} minor"
                else:
                    key_full = f"{key_raw} major"
                
                genre = meta.get('genre', 'Instrumental')
                
                evaluate_tracks.append({
                    'track_id': line_num,
                    'description': description,
                    'prompt': description,
                    'genre': genre,
                    'bpm': meta.get('bpm', 120.0),
                    'key': key_full,
                    'path': data.get('path', ''),
                })
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                continue

    print(f"📊 Found {len(evaluate_tracks)} evaluation tracks")

    if num_samples is None:
        sampled_tracks = evaluate_tracks[start_from:]
    else:
        sampled_tracks = random.sample(evaluate_tracks, min(num_samples, len(evaluate_tracks)))

    test_cases = []
    for track in sampled_tracks:
        test_cases.append({
            'prompt': track['prompt'],
            'expected_style': track['genre'],
            'expected_bpm': track['bpm'],
            'expected_key': track['key'],
            'track_id': track['track_id'],
        })

    print(f"✅ Loaded {len(test_cases)} test cases")
    return test_cases

def analyze_audio(audio_path, sample_rate=32000, target_bpm=None):
    """Analyze audio for BPM, key, and spectral features using Essentia"""
    try:
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=sample_rate)
        audio = loader()

        audio_info = es.AudioLoader(filename=str(audio_path))
        audio_stereo, sr, channels, _, _, _ = audio_info()

        # BPM Detection
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        if target_bpm is not None:
            ratio = target_bpm / bpm
            if 1.7 < ratio < 2.3:
                bpm = bpm * 2
            elif 0.4 < ratio < 0.6:
                bpm = bpm / 2

        # Key Detection
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        if scale == 'minor':
            detected_key_full = f"{key} minor"
        else:
            detected_key_full = f"{key} major"

        # Spectral features
        import librosa
        y, sr_lib = librosa.load(str(audio_path), sr=sample_rate)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_lib))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        rms_energy = np.mean(librosa.feature.rms(y=y))

        return {
            'bpm': float(bpm),
            'key': detected_key_full,
            'key_strength': float(strength),
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zero_crossing_rate),
            'rms_energy': float(rms_energy),
        }

    except Exception as e:
        print(f"      ❌ Error analyzing: {e}")
        return None

def generate_music(model, prompt, duration=15, model_name="model", apply_postprocessing=True, target_key=None, target_bpm=None):
    """Generate music with MusicGen, optionally with post-processing"""
    try:
        start_time = time.time()
        
        model_sample_rate = model.sample_rate
        print(f"      📊 Model sample rate: {model_sample_rate}Hz")
        
        if model_sample_rate != 32000:
            print(f"      ⚠️  WARNING: Model sample rate is {model_sample_rate}Hz, expected 32kHz!")

        model.set_generation_params(
            duration=duration,
            temperature=0.85,
            cfg_coef=5.0,
            top_k=250,
            top_p=0.95,
        )

        with torch.no_grad():
            wav = model.generate([prompt], progress=False)

        generation_time = time.time() - start_time
        
        if wav is not None:
            print(f"      📊 Generated audio shape: {wav.shape}, sample_rate: {model_sample_rate}Hz")
            print(f"      📊 Audio duration: {wav.shape[-1] / model_sample_rate:.2f}s")
        
        if wav is None or wav.numel() == 0:
            print(f"      ⚠️  Generated audio is empty")
            return None, None, 0
        
        if torch.isnan(wav).any() or torch.isinf(wav).any():
            print(f"      ⚠️  Generated audio contains NaN/Inf values, clipping...")
            wav = torch.clamp(wav, -1.0, 1.0)
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        
        audio_min = wav.min().item()
        audio_max = wav.max().item()
        print(f"      📊 Audio range: [{audio_min:.4f}, {audio_max:.4f}]")
        if audio_min < -1.0 or audio_max > 1.0:
            print(f"      ⚠️  Audio values out of range, clamping...")
            wav = torch.clamp(wav, -1.0, 1.0)

        if apply_postprocessing and wav is not None:
            try:
                print(f"      🎛️  Applying post-processing (key/BPM correction only, no mastering)...")
                processor = MusicGenPostProcessor(sample_rate=model.sample_rate)
                wav = processor.process(
                    audio=wav,
                    apply_mastering=False,
                    target_key=target_key,
                    target_bpm=target_bpm,
                    return_tensor=True
                )
                print(f"      ✅ Post-processing complete")
            except Exception as e:
                print(f"      ⚠️  Post-processing failed: {e}, using original audio")
                import traceback
                traceback.print_exc()

        if wav is not None:
            print(f"      📊 Final audio shape: {wav.shape}, sample_rate: {model.sample_rate}Hz")
            wav = torch.clamp(wav, -1.0, 1.0)
            
            target_sample_rate = 44100
            if model.sample_rate != target_sample_rate:
                try:
                    print(f"      🎚️  Upsampling from {model.sample_rate}Hz to {target_sample_rate}Hz (CD quality)...")
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        model.sample_rate,
                        target_sample_rate,
                        resampling_method='sinc_interp_kaiser'
                    )
                    wav = resampler(wav)
                    final_sample_rate = target_sample_rate
                    print(f"      ✅ Audio upsampled to {final_sample_rate}Hz (CD quality)")
                except Exception as resample_err:
                    print(f"      ⚠️  Upsampling failed: {resample_err}, using original {model.sample_rate}Hz")
                    final_sample_rate = model.sample_rate
            else:
                final_sample_rate = model.sample_rate

        return wav, final_sample_rate, generation_time

    except Exception as e:
        print(f"      ❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def apply_lora_to_model(model, rank=16, alpha=32):
    """Apply LoRA adapters to model transformer"""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=['out_proj', 'linear1', 'linear2'],
        lora_dropout=0.1,
        bias='none',
    )

    peft_config_dict = {'default': lora_config}
    model.lm.transformer = LoraModel(
        model.lm.transformer,
        peft_config_dict,
        'default'
    )

    return model

def compare_models(start_from=0, skip_pretrained=False):
    """Compare pretrained vs trained LARGE LoRA on real test dataset"""
    print("="*80)
    if start_from > 0:
        print(f"MUSICGEN LARGE: RESUMING EVALUATION FROM TEST {start_from}")
    else:
        print("MUSICGEN LARGE: EVALUATION WITH POST-PROCESSING (Looperman Test Dataset)")
    print("="*80)
    print("🎛️  Post-processing enabled: Key/BPM Correction ONLY (Mastering DISABLED)")
    print("="*80)

    # Setup logging - ensure directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_FILE, 'w')
    
    def log_print(*args, **kwargs):
        """Print to both console and log file"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    all_test_cases = load_test_cases(TEST_DATASET, num_samples=20, start_from=0)
    
    if start_from > 0:
        test_cases = all_test_cases[start_from-1:]
        log_print(f"📋 Resuming from test {start_from}: {len(test_cases)} tests remaining")
    else:
        test_cases = all_test_cases

    pretrained_dir = OUTPUT_DIR / "pretrained_large"
    trained_dir = OUTPUT_DIR / "trained_large_lora_b00102be_looperman"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    trained_dir.mkdir(parents=True, exist_ok=True)

    log_print(f"\n🧪 Testing {len(test_cases)} prompts from real dataset")
    log_print(f"📁 Output directory: {OUTPUT_DIR}")
    log_print(f"📝 Log file: {LOG_FILE}")

    pretrained_results = []
    trained_results = []

    gen_params = {
        'duration': 15,
        'temperature': 0.85,
        'cfg_coef': 5.0,
        'top_k': 250,
        'top_p': 0.95,
    }

    # PHASE 1: PRETRAINED LARGE MODEL
    if skip_pretrained:
        log_print(f"\n{'='*80}")
        log_print("PHASE 1: PRETRAINED LARGE MODEL - SKIPPED (already completed)")
        log_print(f"{'='*80}")
        pretrained_results = []
        for idx in range(1, len(all_test_cases) + 1):
            pretrained_file = pretrained_dir / f"test_{idx:02d}.wav"
            if pretrained_file.exists():
                pretrained_results.append({
                    'test_id': idx,
                    'test_case': all_test_cases[idx-1],
                    'file': pretrained_file,
                    'analysis': None,
                    'generation_time': 0
                })
        log_print(f"  ✅ Loaded {len(pretrained_results)} existing pretrained results")
    else:
        log_print(f"\n{'='*80}")
        log_print("PHASE 1: PRETRAINED LARGE MODEL")
        log_print(f"{'='*80}")
        log_print(f"  🔵 Loading {PRETRAINED_MODEL}...")

        try:
            pretrained_model = MusicGen.get_pretrained(PRETRAINED_MODEL, device='cuda')
            pretrained_model.set_generation_params(**gen_params)
            log_print("     ✅ Pretrained LARGE model loaded")
        except Exception as e:
            log_print(f"     ❌ Failed to load pretrained model: {e}")
            log_file.close()
            return [], []

        total_tests = len(all_test_cases)
        for idx, test_case in enumerate(all_test_cases, 1):
            log_print(f"\n  Test {idx}/{total_tests}: {test_case['expected_style']}")
            log_print(f"  📝 {test_case['prompt'][:60]}...")
            log_print(f"  🎯 Expected: BPM={test_case['expected_bpm']:.0f}, Key={test_case['expected_key']}")

            expected_key_formatted = test_case['expected_key'].replace(" major", "+").replace(" minor", "-")
            
            wav, sr, gen_time = generate_music(
                pretrained_model, test_case["prompt"],
                duration=15, model_name="pretrained",
                apply_postprocessing=True,
                target_key=expected_key_formatted,
                target_bpm=test_case['expected_bpm']
            )

            pretrained_analysis = None
            pretrained_file = None

            if wav is not None:
                pretrained_file = pretrained_dir / f"test_{idx:02d}.wav"

                try:
                    wav_cpu = wav.cpu().squeeze(0)
                    log_print(f"      💾 Saving pretrained audio to: {pretrained_file}")
                    torchaudio.save(str(pretrained_file), wav_cpu, sr, encoding='PCM_S', bits_per_sample=16)
                    
                    import os
                    file_size = os.path.getsize(str(pretrained_file))
                    log_print(f"      ✅ Saved: {pretrained_file} (size: {file_size} bytes)")

                    pretrained_analysis = analyze_audio(str(pretrained_file), target_bpm=test_case['expected_bpm'])
                    if pretrained_analysis:
                        bpm_error = abs(pretrained_analysis['bpm'] - test_case['expected_bpm'])
                        key_match = keys_match(pretrained_analysis['key'], test_case['expected_key'])
                        status = "✅" if (bpm_error < 5 and key_match) else "⚠️"
                        log_print(f"      🔵 Detected: BPM={pretrained_analysis['bpm']:.1f}, Key={pretrained_analysis['key']} {status}")
                except Exception as e:
                    log_print(f"      ❌ Failed to save/analyze: {e}")
                    import traceback
                    traceback.print_exc()

            pretrained_results.append({
                'test_id': idx,
                'test_case': test_case,
                'file': pretrained_file,
                'analysis': pretrained_analysis,
                'generation_time': gen_time
            })

            clear_gpu_memory()

        if not skip_pretrained:
            log_print(f"\n  🗑️  Unloading pretrained model...")
            del pretrained_model
            clear_gpu_memory()

    # PHASE 2: TRAINED LARGE LoRA MODEL
    log_print(f"\n{'='*80}")
    log_print("PHASE 2: TRAINED LARGE LoRA MODEL")
    log_print(f"{'='*80}")
    
    log_print(f"  🧹 Clearing GPU memory...")
    clear_gpu_memory()
    
    log_print(f"  🟢 Loading trained LARGE LoRA model...")

    try:
        trained_model = MusicGen.get_pretrained(PRETRAINED_MODEL, device='cuda')

        log_print(f"  📦 Loading checkpoint from: {TRAINED_CHECKPOINT}")
        
        checkpoint_path = TRAINED_CHECKPOINT
        checkpoint = None
        checkpoint_file = None
        
        if checkpoint_path.is_dir():
            possible_paths = [
                checkpoint_path / 'checkpoint_15.th',
                checkpoint_path / 'checkpoint_10.th',
                checkpoint_path / 'checkpoint.th',
            ]
            for path in possible_paths:
                if path.exists():
                    checkpoint_file = path
                    log_print(f"  📦 Found checkpoint: {checkpoint_file}")
                    break
            
            if checkpoint_file is None:
                try:
                    sys.path.insert(0, '/home/ubuntu/beats/musicgen/training/audiocraft')
                    from audiocraft.utils import checkpoint as ckpt_utils
                    xp_sig = checkpoint_path.name
                    resolved = ckpt_utils.resolve_checkpoint_path(f'//sig/{xp_sig}')
                    if resolved and resolved.exists():
                        checkpoint_file = resolved
                        log_print(f"  📦 Resolved checkpoint: {checkpoint_file}")
                except Exception as e:
                    log_print(f"  ⚠️  Could not resolve checkpoint: {e}")
            
            if checkpoint_file is None:
                log_print(f"  ⚠️  No checkpoint file found in {checkpoint_path}")
            else:
                log_print(f"  📦 Loading checkpoint to CPU (to avoid OOM)...")
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
        else:
            if checkpoint_path.exists():
                checkpoint_file = checkpoint_path
                log_print(f"  📦 Loading checkpoint to CPU (to avoid OOM)...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            else:
                log_print(f"  ⚠️  Checkpoint file not found: {checkpoint_path}")

        if checkpoint is not None:
            log_print(f"  📦 Moving checkpoint to GPU (incremental to avoid OOM)...")
            state_dict = None
            if 'best_state' in checkpoint and 'model' in checkpoint['best_state']:
                log_print(f"  📦 Loading from best_state...")
                state_dict = checkpoint['best_state']['model']
            elif 'state' in checkpoint and 'model' in checkpoint['state']:
                log_print(f"  📦 Loading from state...")
                state_dict = checkpoint['state']['model']
            elif 'model' in checkpoint:
                log_print(f"  📦 Loading from model key...")
                state_dict = checkpoint['model']
            else:
                log_print(f"  ⚠️  Checkpoint format unclear, trying direct load...")
                state_dict = checkpoint
            
            if state_dict is not None:
                lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                
                if lora_keys:
                    log_print(f"  📦 Found {len(lora_keys)} LoRA adapter keys in checkpoint")
                    log_print(f"  📌 Applying LoRA adapters to match checkpoint structure...")
                    trained_model = apply_lora_to_model(trained_model, rank=16, alpha=32)
                    
                    log_print(f"  📦 Loading only LoRA adapter weights...")
                    filtered_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k.lower()}
                    state_dict = filtered_state_dict
                else:
                    log_print(f"  ⚠️  No LoRA keys found - checkpoint contains FULL model weights")
                    log_print(f"  📦 Loading full model weights (NOT applying LoRA adapters)...")
                
                state_dict_gpu = {}
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        state_dict_gpu[k] = v.cuda()
                    else:
                        state_dict_gpu[k] = v
                
                missing, unexpected = trained_model.lm.load_state_dict(state_dict_gpu, strict=False)
                if missing:
                    log_print(f"  ⚠️  Missing keys: {len(missing)}")
                    if len(missing) > 100:
                        log_print(f"      (showing first 10: {list(missing)[:10]})")
                if unexpected:
                    log_print(f"  ⚠️  Unexpected keys: {len(unexpected)}")
                    if len(unexpected) > 100:
                        log_print(f"      (showing first 10: {list(unexpected)[:10]})")
                
                del state_dict_gpu
                clear_gpu_memory()
            
            del checkpoint
            clear_gpu_memory()
        else:
            log_print(f"  ⚠️  Using model as-is (checkpoint not available)")

        trained_model.lm.eval()
        trained_model.set_generation_params(**gen_params)
        log_print("     ✅ Trained LARGE LoRA model loaded")

    except Exception as e:
        log_print(f"     ❌ Failed to load trained model: {e}")
        import traceback
        traceback.print_exc()
        log_file.close()
        return pretrained_results, []

    total_tests = len(all_test_cases)
    start_idx = start_from if start_from > 0 else 1
    for idx, test_case in enumerate(test_cases, start_idx):
        log_print(f"\n  Test {idx}/{total_tests}: {test_case['expected_style']}")
        log_print(f"  📝 {test_case['prompt'][:60]}...")
        log_print(f"  🎯 Expected: BPM={test_case['expected_bpm']:.0f}, Key={test_case['expected_key']}")

        expected_key_formatted = test_case['expected_key'].replace(" major", "+").replace(" minor", "-")
        
        wav, sr, gen_time = generate_music(
            trained_model, test_case["prompt"],
            duration=15, model_name="trained",
            apply_postprocessing=True,
            target_key=expected_key_formatted,
            target_bpm=test_case['expected_bpm']
        )

        trained_analysis = None
        trained_file = None

        if wav is not None:
            trained_file = trained_dir / f"test_{idx:02d}.wav"

            try:
                wav_cpu = wav.cpu()
                if wav_cpu.dim() == 3:
                    wav_cpu = wav_cpu.squeeze(0)
                elif wav_cpu.dim() == 1:
                    wav_cpu = wav_cpu.unsqueeze(0)
                
                wav_cpu = torch.clamp(wav_cpu, -1.0, 1.0)
                wav_cpu = torch.nan_to_num(wav_cpu, nan=0.0, posinf=1.0, neginf=-1.0)
                
                log_print(f"      💾 Saving trained audio to: {trained_file}")
                torchaudio.save(str(trained_file), wav_cpu, sr, encoding='PCM_S', bits_per_sample=16)
                
                import os
                file_size = os.path.getsize(str(trained_file))
                log_print(f"      ✅ Saved: {trained_file} (size: {file_size} bytes)")
                if file_size == 0:
                    log_print(f"      ❌ WARNING: File was created but is 0 bytes! Something went wrong.")

                trained_analysis = analyze_audio(str(trained_file), target_bpm=test_case['expected_bpm'])
                if trained_analysis:
                    bpm_error = abs(trained_analysis['bpm'] - test_case['expected_bpm'])
                    key_match = keys_match(trained_analysis['key'], test_case['expected_key'])
                    status = "✅" if (bpm_error < 5 and key_match) else "⚠️"
                    log_print(f"      🟢 Detected: BPM={trained_analysis['bpm']:.1f}, Key={trained_analysis['key']} {status}")
            except Exception as e:
                log_print(f"      ❌ Failed: {e}")
                import traceback
                traceback.print_exc()

        trained_results.append({
            'test_id': idx,
            'test_case': test_case,
            'file': trained_file,
            'analysis': trained_analysis,
            'generation_time': gen_time
        })

        clear_gpu_memory()

    log_print(f"\n  🗑️  Unloading trained model...")
    del trained_model
    clear_gpu_memory()

    # PHASE 3: ANALYSIS
    log_print(f"\n{'='*80}")
    log_print("📊 FINAL ANALYSIS")
    log_print(f"{'='*80}")

    bpm_errors_pretrained = []
    bpm_errors_trained = []
    key_matches_pretrained = []
    key_matches_trained = []
    style_matches_pretrained = []
    style_matches_trained = []

    style_profiles = {
        'Dark Trap': {'bpm_range': (130, 165), 'sc_range': (1500, 3500), 'zc_range': (0.05, 0.15)},
        'Drill': {'bpm_range': (140, 170), 'sc_range': (1500, 3000), 'zc_range': (0.04, 0.10)},
        'AfroBeatz': {'bpm_range': (100, 130), 'sc_range': (2000, 4000), 'zc_range': (0.05, 0.12)},
        'Beast On Beatz': {'bpm_range': (80, 145), 'sc_range': (2000, 4000), 'zc_range': (0.06, 0.15)},
        'Rowezart': {'bpm_range': (60, 150), 'sc_range': (2000, 4500), 'zc_range': (0.06, 0.14)},
        'Instrumental': {'bpm_range': (60, 180), 'sc_range': (1000, 5000), 'zc_range': (0.03, 0.20)},
    }

    def calculate_style_match(analysis, expected_style):
        """Calculate style match score"""
        profile = style_profiles.get(expected_style, style_profiles['Instrumental'])
        scores = []

        bpm_min, bpm_max = profile['bpm_range']
        if bpm_min <= analysis['bpm'] <= bpm_max:
            scores.append(1.0)
        else:
            if analysis['bpm'] < bpm_min:
                scores.append(max(0, 1 - (bpm_min - analysis['bpm']) / bpm_min))
            else:
                scores.append(max(0, 1 - (analysis['bpm'] - bpm_max) / bpm_max))

        sc_min, sc_max = profile['sc_range']
        if sc_min <= analysis['spectral_centroid'] <= sc_max:
            scores.append(1.0)
        else:
            if analysis['spectral_centroid'] < sc_min:
                scores.append(max(0, 1 - (sc_min - analysis['spectral_centroid']) / sc_min))
            else:
                scores.append(max(0, 1 - (analysis['spectral_centroid'] - sc_max) / sc_max))

        zc_min, zc_max = profile['zc_range']
        if zc_min <= analysis['zero_crossing_rate'] <= zc_max:
            scores.append(1.0)
        else:
            if analysis['zero_crossing_rate'] < zc_min:
                scores.append(max(0, 1 - (zc_min - analysis['zero_crossing_rate']) / zc_min))
            else:
                scores.append(max(0, 1 - (analysis['zero_crossing_rate'] - zc_max) / zc_max))

        return np.mean(scores)

    # Match results by test_id
    pretrained_dict = {r['test_id']: r for r in pretrained_results}
    trained_dict = {r['test_id']: r for r in trained_results}
    
    all_test_ids = sorted(set(pretrained_dict.keys()) & set(trained_dict.keys()))
    
    log_print(f"\n📊 Comparing {len(all_test_ids)} test cases with both pretrained and trained results...")
    
    for test_id in all_test_ids:
        pretrained_data = pretrained_dict[test_id]
        trained_data = trained_dict[test_id]
        test_case = pretrained_data['test_case']

        pretrained_analysis = pretrained_data['analysis']
        trained_analysis = trained_data['analysis']

        if pretrained_analysis:
            bpm_error = abs(pretrained_analysis['bpm'] - test_case['expected_bpm'])
            bpm_errors_pretrained.append(bpm_error)
            key_match = keys_match(pretrained_analysis['key'], test_case['expected_key'])
            key_matches_pretrained.append(1 if key_match else 0)
            style_match = calculate_style_match(pretrained_analysis, test_case['expected_style'])
            style_matches_pretrained.append(style_match)

        if trained_analysis:
            bpm_error = abs(trained_analysis['bpm'] - test_case['expected_bpm'])
            bpm_errors_trained.append(bpm_error)
            key_match = keys_match(trained_analysis['key'], test_case['expected_key'])
            key_matches_trained.append(1 if key_match else 0)
            style_match = calculate_style_match(trained_analysis, test_case['expected_style'])
            style_matches_trained.append(style_match)

    # Calculate metrics
    if bpm_errors_pretrained and bpm_errors_trained:
        pretrained_avg_bpm_error = np.mean(bpm_errors_pretrained)
        trained_avg_bpm_error = np.mean(bpm_errors_trained)

        pretrained_key_accuracy = np.mean(key_matches_pretrained) * 100
        trained_key_accuracy = np.mean(key_matches_trained) * 100

        pretrained_avg_style = np.mean(style_matches_pretrained) * 100
        trained_avg_style = np.mean(style_matches_trained) * 100

        pretrained_avg_time = np.mean([r['generation_time'] for r in pretrained_results])
        trained_avg_time = np.mean([r['generation_time'] for r in trained_results])

        pretrained_wins = 0
        trained_wins = 0

        for i in range(len(bpm_errors_pretrained)):
            if bpm_errors_trained[i] < bpm_errors_pretrained[i]:
                trained_wins += 1
            elif bpm_errors_trained[i] > bpm_errors_pretrained[i]:
                pretrained_wins += 1

            if style_matches_trained[i] > style_matches_pretrained[i]:
                trained_wins += 1
            elif style_matches_trained[i] < style_matches_pretrained[i]:
                pretrained_wins += 1

            if key_matches_trained[i] and not key_matches_pretrained[i]:
                trained_wins += 1
            elif key_matches_pretrained[i] and not key_matches_trained[i]:
                pretrained_wins += 1

        log_print(f"\n🎯 BPM Accuracy:")
        log_print(f"  Pretrained LARGE Average Error: ±{pretrained_avg_bpm_error:.1f} BPM")
        log_print(f"  Trained LARGE LoRA Average Error: ±{trained_avg_bpm_error:.1f} BPM")
        if trained_avg_bpm_error < pretrained_avg_bpm_error:
            improvement = pretrained_avg_bpm_error - trained_avg_bpm_error
            log_print(f"  🟢 Trained LoRA is {improvement:.1f} BPM more accurate on average")
        else:
            diff = trained_avg_bpm_error - pretrained_avg_bpm_error
            log_print(f"  🔵 Pretrained is {diff:.1f} BPM more accurate on average")

        log_print(f"\n🎨 Style Matching:")
        log_print(f"  Pretrained LARGE Average: {pretrained_avg_style:.1f}%")
        log_print(f"  Trained LARGE LoRA Average: {trained_avg_style:.1f}%")
        if trained_avg_style > pretrained_avg_style:
            improvement = trained_avg_style - pretrained_avg_style
            log_print(f"  🟢 Trained LoRA is {improvement:.1f}% better at style matching")
        else:
            diff = pretrained_avg_style - trained_avg_style
            log_print(f"  🔵 Pretrained is {diff:.1f}% better at style matching")

        log_print(f"\n🎹 Key Accuracy:")
        log_print(f"  Pretrained LARGE: {sum(key_matches_pretrained)}/{len(key_matches_pretrained)} correct ({pretrained_key_accuracy:.1f}%)")
        log_print(f"  Trained LARGE LoRA: {sum(key_matches_trained)}/{len(key_matches_trained)} correct ({trained_key_accuracy:.1f}%)")
        if trained_key_accuracy > pretrained_key_accuracy:
            improvement = trained_key_accuracy - pretrained_key_accuracy
            log_print(f"  🟢 Trained LoRA is {improvement:.1f}% more accurate")
        elif trained_key_accuracy < pretrained_key_accuracy:
            diff = pretrained_key_accuracy - trained_key_accuracy
            log_print(f"  🔵 Pretrained is {diff:.1f}% more accurate")
        else:
            log_print(f"  ⚖️  Both models have the same key accuracy")

        log_print(f"\n⏱️  Generation Speed:")
        log_print(f"  Pretrained LARGE Average: {pretrained_avg_time:.1f}s")
        log_print(f"  Trained LARGE LoRA Average: {trained_avg_time:.1f}s")

        log_print(f"\n🏆 WINNER:")
        log_print(f"  🔵 Pretrained LARGE Wins: {pretrained_wins}")
        log_print(f"  🟢 Trained LARGE LoRA Wins: {trained_wins}")
        log_print()

        if trained_wins > pretrained_wins:
            log_print(f"  🎉 TRAINED LARGE LoRA MODEL PERFORMED BETTER! 🎉")
        elif pretrained_wins > trained_wins:
            log_print(f"  🔵 Pretrained LARGE model performed better")
        else:
            log_print(f"  ⚖️  Both models performed equally")

        # Save detailed report
        report = {
            'model_comparison': 'Pretrained LARGE vs Trained LARGE LoRA',
            'summary': {
                'pretrained_avg_bpm_error': float(pretrained_avg_bpm_error),
                'trained_avg_bpm_error': float(trained_avg_bpm_error),
                'pretrained_key_accuracy': float(pretrained_key_accuracy),
                'trained_key_accuracy': float(trained_key_accuracy),
                'pretrained_avg_style': float(pretrained_avg_style),
                'trained_avg_style': float(trained_avg_style),
                'pretrained_wins': int(pretrained_wins),
                'trained_wins': int(trained_wins),
            },
            'test_results': [
                {
                    'test_id': test_id,
                    'prompt': pretrained_dict[test_id]['test_case']['prompt'],
                    'expected_bpm': pretrained_dict[test_id]['test_case']['expected_bpm'],
                    'expected_key': pretrained_dict[test_id]['test_case']['expected_key'],
                    'expected_style': pretrained_dict[test_id]['test_case']['expected_style'],
                    'pretrained_bpm': pretrained_dict[test_id]['analysis']['bpm'] if pretrained_dict[test_id]['analysis'] else None,
                    'pretrained_key': pretrained_dict[test_id]['analysis']['key'] if pretrained_dict[test_id]['analysis'] else None,
                    'trained_bpm': trained_dict[test_id]['analysis']['bpm'] if trained_dict[test_id]['analysis'] else None,
                    'trained_key': trained_dict[test_id]['analysis']['key'] if trained_dict[test_id]['analysis'] else None,
                }
                for test_id in all_test_ids
            ]
        }

        report_file = OUTPUT_DIR / "large_lora_b00102be_looperman_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        log_print(f"\n💾 Detailed report: {report_file}")
        log_print(f"🎵 Audio samples:")
        log_print(f"  🔵 Pretrained LARGE: {pretrained_dir}/")
        log_print(f"  🟢 Trained LARGE LoRA (checkpoint b00102be, Looperman test): {trained_dir}/")

    log_print(f"\n{'='*80}")
    log_print("✅ TEST COMPLETE")
    log_print(f"{'='*80}\n")
    log_print(f"📝 Full log saved to: {LOG_FILE}")

    log_file.close()
    return pretrained_results, trained_results

if __name__ == "__main__":
    import sys
    
    random.seed(42)
    
    start_from = 0
    skip_pretrained = False
    
    if len(sys.argv) > 1:
        try:
            start_from = int(sys.argv[1])
            print(f"🔄 Resuming from test {start_from}")
        except ValueError:
            print(f"⚠️  Invalid start_from value: {sys.argv[1]}, starting from beginning")
    
    from pathlib import Path
    pretrained_dir = Path("/home/ubuntu/beats/musicgen/performance/looperman/pretrained_large")
    pretrained_count = len(list(pretrained_dir.glob("test_*.wav"))) if pretrained_dir.exists() else 0
    skip_pretrained = False  # Always run pretrained tests
    
    try:
        pretrained_results, trained_results = compare_models(start_from=start_from, skip_pretrained=skip_pretrained)
        print("🎉 Test completed successfully!")
        print(f"📝 Check log file: {LOG_FILE}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

