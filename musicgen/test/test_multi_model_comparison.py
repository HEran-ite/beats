#!/usr/bin/env python3
"""
Multi-Model Comparison Test: Pretrained vs e51bc38f vs b00102be
Tests 500 samples from looperman_audio dataset
Compares:
1. Pretrained LARGE model
2. e51bc38f (trained from checkpoint 983190cf)
3. b00102be (trained from scratch)
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
CHECKPOINT_E51BC38F = Path("/home/ubuntu/amanuel/beats/audiocraft_checkpoints/xps/e51bc38f")
CHECKPOINT_B00102BE = Path("/home/ubuntu/amanuel/beats/audiocraft_checkpoints/xps/b00102be")
TEST_DATASET = "/home/ubuntu/beats/musicgen/training/audiocraft/egs/looperman_audio/evaluate"
OUTPUT_DIR = Path("/home/ubuntu/beats/musicgen/performance/looperman_500")
LOG_FILE = OUTPUT_DIR / "test_multi_model_500.log"

# Model configurations
MODELS = {
    'pretrained': {
        'name': 'Pretrained LARGE',
        'checkpoint': None,
        'base_model': PRETRAINED_MODEL,
        'output_dir': 'pretrained_large',
        'color': '🔵',
    },
    'e51bc38f': {
        'name': 'e51bc38f (from 983190cf)',
        'checkpoint': CHECKPOINT_E51BC38F,
        'base_model': PRETRAINED_MODEL,
        'output_dir': 'trained_e51bc38f_looperman',
        'color': '🟢',
    },
    'b00102be': {
        'name': 'b00102be (from scratch)',
        'checkpoint': CHECKPOINT_B00102BE,
        'base_model': PRETRAINED_MODEL,
        'output_dir': 'trained_b00102be_looperman',
        'color': '🟡',
    },
}

def keys_match(key1: str, key2: str) -> bool:
    """Check if two keys match, considering enharmonic equivalents"""
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

def load_test_cases(dataset_path, num_samples=500, start_from=0):
    """Load test cases from dataset (JSONL format)"""
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
        
        if wav is None or wav.numel() == 0:
            print(f"      ⚠️  Generated audio is empty")
            return None, None, 0
        
        if torch.isnan(wav).any() or torch.isinf(wav).any():
            wav = torch.clamp(wav, -1.0, 1.0)
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        
        audio_min = wav.min().item()
        audio_max = wav.max().item()
        if audio_min < -1.0 or audio_max > 1.0:
            wav = torch.clamp(wav, -1.0, 1.0)

        if apply_postprocessing and wav is not None:
            try:
                processor = MusicGenPostProcessor(sample_rate=model.sample_rate)
                wav = processor.process(
                    audio=wav,
                    apply_mastering=False,
                    target_key=target_key,
                    target_bpm=target_bpm,
                    return_tensor=True
                )
            except Exception as e:
                print(f"      ⚠️  Post-processing failed: {e}, using original audio")

        if wav is not None:
            wav = torch.clamp(wav, -1.0, 1.0)
            
            target_sample_rate = 44100
            if model.sample_rate != target_sample_rate:
                try:
                    resampler = torchaudio.transforms.Resample(
                        model.sample_rate,
                        target_sample_rate,
                        resampling_method='sinc_interp_kaiser'
                    )
                    wav = resampler(wav)
                    final_sample_rate = target_sample_rate
                except Exception as resample_err:
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

def load_checkpoint_model(model_key, model_config, log_print):
    """Load a model with its checkpoint if applicable"""
    log_print(f"  {model_config['color']} Loading {model_config['name']}...")
    
    try:
        model = MusicGen.get_pretrained(model_config['base_model'], device='cuda')
        
        if model_config['checkpoint'] is not None:
            checkpoint_path = model_config['checkpoint']
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
                        log_print(f"    📦 Found checkpoint: {checkpoint_file}")
                        break
                
                if checkpoint_file is None:
                    try:
                        sys.path.insert(0, '/home/ubuntu/beats/musicgen/training/audiocraft')
                        from audiocraft.utils import checkpoint as ckpt_utils
                        xp_sig = checkpoint_path.name
                        resolved = ckpt_utils.resolve_checkpoint_path(f'//sig/{xp_sig}')
                        if resolved and resolved.exists():
                            checkpoint_file = resolved
                            log_print(f"    📦 Resolved checkpoint: {checkpoint_file}")
                    except Exception as e:
                        log_print(f"    ⚠️  Could not resolve checkpoint: {e}")
                
                if checkpoint_file is None:
                    log_print(f"    ⚠️  No checkpoint file found in {checkpoint_path}")
                else:
                    log_print(f"    📦 Loading checkpoint to CPU...")
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
            else:
                if checkpoint_path.exists():
                    checkpoint_file = checkpoint_path
                    log_print(f"    📦 Loading checkpoint to CPU...")
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                else:
                    log_print(f"    ⚠️  Checkpoint file not found: {checkpoint_path}")

            if checkpoint is not None:
                state_dict = None
                if 'best_state' in checkpoint and 'model' in checkpoint['best_state']:
                    state_dict = checkpoint['best_state']['model']
                elif 'state' in checkpoint and 'model' in checkpoint['state']:
                    state_dict = checkpoint['state']['model']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                if state_dict is not None:
                    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                    
                    if lora_keys:
                        log_print(f"    📦 Found {len(lora_keys)} LoRA adapter keys")
                        model = apply_lora_to_model(model, rank=16, alpha=32)
                        filtered_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k.lower()}
                        state_dict = filtered_state_dict
                    else:
                        log_print(f"    📦 Loading full model weights")
                    
                    state_dict_gpu = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            state_dict_gpu[k] = v.cuda()
                        else:
                            state_dict_gpu[k] = v
                    
                    missing, unexpected = model.lm.load_state_dict(state_dict_gpu, strict=False)
                    if missing:
                        log_print(f"    ⚠️  Missing keys: {len(missing)}")
                    if unexpected:
                        log_print(f"    ⚠️  Unexpected keys: {len(unexpected)}")
                    
                    del state_dict_gpu
                    clear_gpu_memory()
                
                del checkpoint
                clear_gpu_memory()
        
        model.lm.eval()
        model.set_generation_params(
            duration=15,
            temperature=0.85,
            cfg_coef=5.0,
            top_k=250,
            top_p=0.95,
        )
        log_print(f"    ✅ {model_config['name']} loaded successfully")
        return model
        
    except Exception as e:
        log_print(f"    ❌ Failed to load {model_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model(model_key, model_config, test_cases, output_dir, log_print, start_from=0):
    """Test a single model on all test cases"""
    model_results = []
    model_dir = output_dir / model_config['output_dir']
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_checkpoint_model(model_key, model_config, log_print)
    if model is None:
        return []
    
    total_tests = len(test_cases)
    for idx, test_case in enumerate(test_cases, 1):
        if idx < start_from:
            continue
            
        log_print(f"\n  Test {idx}/{total_tests} ({model_config['color']} {model_config['name']}): {test_case['expected_style']}")
        log_print(f"  📝 {test_case['prompt'][:60]}...")
        log_print(f"  🎯 Expected: BPM={test_case['expected_bpm']:.0f}, Key={test_case['expected_key']}")

        expected_key_formatted = test_case['expected_key'].replace(" major", "+").replace(" minor", "-")
        
        wav, sr, gen_time = generate_music(
            model, test_case["prompt"],
            duration=15, model_name=model_key,
            apply_postprocessing=True,
            target_key=expected_key_formatted,
            target_bpm=test_case['expected_bpm']
        )

        analysis = None
        audio_file = None

        if wav is not None:
            audio_file = model_dir / f"test_{idx:05d}.wav"

            try:
                wav_cpu = wav.cpu()
                if wav_cpu.dim() == 3:
                    wav_cpu = wav_cpu.squeeze(0)
                elif wav_cpu.dim() == 1:
                    wav_cpu = wav_cpu.unsqueeze(0)
                
                wav_cpu = torch.clamp(wav_cpu, -1.0, 1.0)
                wav_cpu = torch.nan_to_num(wav_cpu, nan=0.0, posinf=1.0, neginf=-1.0)
                
                torchaudio.save(str(audio_file), wav_cpu, sr, encoding='PCM_S', bits_per_sample=16)
                
                import os
                file_size = os.path.getsize(str(audio_file))
                if file_size == 0:
                    log_print(f"      ❌ WARNING: File is 0 bytes!")

                analysis = analyze_audio(str(audio_file), target_bpm=test_case['expected_bpm'])
                if analysis:
                    bpm_error = abs(analysis['bpm'] - test_case['expected_bpm'])
                    key_match = keys_match(analysis['key'], test_case['expected_key'])
                    status = "✅" if (bpm_error < 5 and key_match) else "⚠️"
                    log_print(f"      {model_config['color']} Detected: BPM={analysis['bpm']:.1f}, Key={analysis['key']} {status}")
            except Exception as e:
                log_print(f"      ❌ Failed: {e}")

        model_results.append({
            'test_id': idx,
            'test_case': test_case,
            'file': audio_file,
            'analysis': analysis,
            'generation_time': gen_time
        })

        if idx % 10 == 0:
            log_print(f"  📊 Progress: {idx}/{total_tests} tests completed")
            clear_gpu_memory()

    log_print(f"\n  🗑️  Unloading {model_config['name']}...")
    del model
    clear_gpu_memory()
    
    return model_results

def compare_all_models(start_from=0, num_samples=500):
    """Compare all three models on test dataset"""
    print("="*80)
    print("MUSICGEN MULTI-MODEL COMPARISON: 500 SAMPLES")
    print("="*80)
    print("Models:")
    print("  🔵 Pretrained LARGE")
    print("  🟢 e51bc38f (trained from checkpoint 983190cf)")
    print("  🟡 b00102be (trained from scratch)")
    print("="*80)

    # Setup logging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_FILE, 'w')
    
    def log_print(*args, **kwargs):
        """Print to both console and log file"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    all_test_cases = load_test_cases(TEST_DATASET, num_samples=num_samples, start_from=0)
    
    if start_from > 0:
        test_cases = all_test_cases[start_from-1:]
        log_print(f"📋 Resuming from test {start_from}: {len(test_cases)} tests remaining")
    else:
        test_cases = all_test_cases

    log_print(f"\n🧪 Testing {len(test_cases)} prompts from real dataset")
    log_print(f"📁 Output directory: {OUTPUT_DIR}")
    log_print(f"📝 Log file: {LOG_FILE}")

    all_results = {}
    
    # Test each model
    for model_key, model_config in MODELS.items():
        log_print(f"\n{'='*80}")
        log_print(f"TESTING: {model_config['name']}")
        log_print(f"{'='*80}")
        
        results = test_model(model_key, model_config, test_cases, OUTPUT_DIR, log_print, start_from=start_from)
        all_results[model_key] = results
        
        log_print(f"✅ Completed {len(results)} tests for {model_config['name']}")

    # Final analysis
    log_print(f"\n{'='*80}")
    log_print("📊 COMPREHENSIVE FINAL ANALYSIS")
    log_print(f"{'='*80}")

    # Style profiles for style matching
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

    # Calculate metrics for each model
    metrics = {}
    for model_key, results in all_results.items():
        model_config = MODELS[model_key]
        bpm_errors = []
        key_matches = []
        style_matches = []
        generation_times = []
        perfect_bpm_matches = 0  # Within 5 BPM
        perfect_key_matches = 0
        
        for result in results:
            if result['analysis']:
                bpm_error = abs(result['analysis']['bpm'] - result['test_case']['expected_bpm'])
                bpm_errors.append(bpm_error)
                if bpm_error < 5:
                    perfect_bpm_matches += 1
                
                key_match = keys_match(result['analysis']['key'], result['test_case']['expected_key'])
                key_matches.append(1 if key_match else 0)
                if key_match:
                    perfect_key_matches += 1
                
                style_match = calculate_style_match(result['analysis'], result['test_case']['expected_style'])
                style_matches.append(style_match)
            
            if result['generation_time'] > 0:
                generation_times.append(result['generation_time'])
        
        metrics[model_key] = {
            'name': model_config['name'],
            'color': model_config['color'],
            'avg_bpm_error': np.mean(bpm_errors) if bpm_errors else 0,
            'std_bpm_error': np.std(bpm_errors) if bpm_errors else 0,
            'min_bpm_error': np.min(bpm_errors) if bpm_errors else 0,
            'max_bpm_error': np.max(bpm_errors) if bpm_errors else 0,
            'key_accuracy': np.mean(key_matches) * 100 if key_matches else 0,
            'style_match': np.mean(style_matches) * 100 if style_matches else 0,
            'avg_generation_time': np.mean(generation_times) if generation_times else 0,
            'total_tests': len(results),
            'successful_tests': len([r for r in results if r['analysis']]),
            'perfect_bpm_matches': perfect_bpm_matches,
            'perfect_key_matches': perfect_key_matches,
            'bpm_errors': bpm_errors,
            'key_matches': key_matches,
            'style_matches': style_matches,
        }

    # Print comprehensive comparison
    log_print(f"\n{'='*80}")
    log_print("📊 METRICS SUMMARY")
    log_print(f"{'='*80}")
    
    log_print(f"\n🎯 BPM Accuracy:")
    log_print(f"{'Model':<40} {'Avg Error':<12} {'Std Dev':<12} {'Perfect (<5 BPM)':<15} {'Success Rate':<12}")
    log_print("-" * 100)
    for model_key, metric in metrics.items():
        success_rate = (metric['successful_tests'] / metric['total_tests'] * 100) if metric['total_tests'] > 0 else 0
        log_print(f"{metric['color']} {metric['name']:<38} ±{metric['avg_bpm_error']:.2f} BPM    ±{metric['std_bpm_error']:.2f} BPM    {metric['perfect_bpm_matches']}/{metric['successful_tests']:<13} {success_rate:.1f}%")

    log_print(f"\n🎹 Key Accuracy:")
    log_print(f"{'Model':<40} {'Accuracy':<12} {'Perfect Matches':<15} {'Success Rate':<12}")
    log_print("-" * 100)
    for model_key, metric in metrics.items():
        success_rate = (metric['successful_tests'] / metric['total_tests'] * 100) if metric['total_tests'] > 0 else 0
        log_print(f"{metric['color']} {metric['name']:<38} {metric['key_accuracy']:.1f}%        {metric['perfect_key_matches']}/{metric['successful_tests']:<13} {success_rate:.1f}%")

    log_print(f"\n🎨 Style Matching:")
    log_print(f"{'Model':<40} {'Style Match':<12} {'Success Rate':<12}")
    log_print("-" * 100)
    for model_key, metric in metrics.items():
        success_rate = (metric['successful_tests'] / metric['total_tests'] * 100) if metric['total_tests'] > 0 else 0
        log_print(f"{metric['color']} {metric['name']:<38} {metric['style_match']:.1f}%        {success_rate:.1f}%")

    log_print(f"\n⏱️  Generation Speed:")
    log_print(f"{'Model':<40} {'Avg Time':<12} {'Total Tests':<12}")
    log_print("-" * 100)
    for model_key, metric in metrics.items():
        log_print(f"{metric['color']} {metric['name']:<38} {metric['avg_generation_time']:.2f}s       {metric['total_tests']}")

    # Determine winners
    log_print(f"\n{'='*80}")
    log_print("🏆 WINNER ANALYSIS")
    log_print(f"{'='*80}")
    
    # Find best model for each metric
    best_bpm = min(metrics.items(), key=lambda x: x[1]['avg_bpm_error'])
    best_key = max(metrics.items(), key=lambda x: x[1]['key_accuracy'])
    best_style = max(metrics.items(), key=lambda x: x[1]['style_match'])
    
    log_print(f"\n🥇 Best BPM Accuracy: {best_bpm[1]['color']} {best_bpm[1]['name']} (±{best_bpm[1]['avg_bpm_error']:.2f} BPM)")
    log_print(f"🥇 Best Key Accuracy: {best_key[1]['color']} {best_key[1]['name']} ({best_key[1]['key_accuracy']:.1f}%)")
    log_print(f"🥇 Best Style Matching: {best_style[1]['color']} {best_style[1]['name']} ({best_style[1]['style_match']:.1f}%)")
    
    # Head-to-head comparisons
    log_print(f"\n{'='*80}")
    log_print("⚔️  HEAD-TO-HEAD COMPARISONS")
    log_print(f"{'='*80}")
    
    model_keys = list(metrics.keys())
    for i, model1_key in enumerate(model_keys):
        for model2_key in model_keys[i+1:]:
            model1 = metrics[model1_key]
            model2 = metrics[model2_key]
            
            log_print(f"\n{model1['color']} {model1['name']} vs {model2['color']} {model2['name']}:")
            
            # BPM comparison
            if model1['avg_bpm_error'] < model2['avg_bpm_error']:
                diff = model2['avg_bpm_error'] - model1['avg_bpm_error']
                log_print(f"  🎯 BPM: {model1['color']} {model1['name']} wins by {diff:.2f} BPM (more accurate)")
            elif model2['avg_bpm_error'] < model1['avg_bpm_error']:
                diff = model1['avg_bpm_error'] - model2['avg_bpm_error']
                log_print(f"  🎯 BPM: {model2['color']} {model2['name']} wins by {diff:.2f} BPM (more accurate)")
            else:
                log_print(f"  🎯 BPM: Tie")
            
            # Key comparison
            if model1['key_accuracy'] > model2['key_accuracy']:
                diff = model1['key_accuracy'] - model2['key_accuracy']
                log_print(f"  🎹 Key: {model1['color']} {model1['name']} wins by {diff:.1f}%")
            elif model2['key_accuracy'] > model1['key_accuracy']:
                diff = model2['key_accuracy'] - model1['key_accuracy']
                log_print(f"  🎹 Key: {model2['color']} {model2['name']} wins by {diff:.1f}%")
            else:
                log_print(f"  🎹 Key: Tie")
            
            # Style comparison
            if model1['style_match'] > model2['style_match']:
                diff = model1['style_match'] - model2['style_match']
                log_print(f"  🎨 Style: {model1['color']} {model1['name']} wins by {diff:.1f}%")
            elif model2['style_match'] > model1['style_match']:
                diff = model2['style_match'] - model1['style_match']
                log_print(f"  🎨 Style: {model2['color']} {model2['name']} wins by {diff:.1f}%")
            else:
                log_print(f"  🎨 Style: Tie")
    
    # Overall winner (weighted score)
    log_print(f"\n{'='*80}")
    log_print("🏅 OVERALL WINNER (Weighted Score)")
    log_print(f"{'='*80}")
    
    overall_scores = {}
    for model_key, metric in metrics.items():
        # Normalize scores (lower BPM error is better, higher key/style is better)
        # Weight: BPM 40%, Key 30%, Style 30%
        max_bpm_error = max(m['avg_bpm_error'] for m in metrics.values()) if metrics.values() else 1
        bpm_score = (1 - metric['avg_bpm_error'] / max_bpm_error) * 100 if max_bpm_error > 0 else 0
        
        overall_score = (bpm_score * 0.4) + (metric['key_accuracy'] * 0.3) + (metric['style_match'] * 0.3)
        overall_scores[model_key] = overall_score
    
    sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    log_print(f"\n{'Rank':<6} {'Model':<40} {'Overall Score':<15} {'Breakdown'}")
    log_print("-" * 100)
    for rank, (model_key, score) in enumerate(sorted_models, 1):
        metric = metrics[model_key]
        breakdown = f"BPM:{metric['avg_bpm_error']:.1f} Key:{metric['key_accuracy']:.1f}% Style:{metric['style_match']:.1f}%"
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        log_print(f"{medal} {rank:<4} {metric['color']} {metric['name']:<38} {score:.2f}            {breakdown}")
    
    winner_key = sorted_models[0][0]
    winner = metrics[winner_key]
    log_print(f"\n🎉 OVERALL WINNER: {winner['color']} {winner['name']} with score {sorted_models[0][1]:.2f}!")

    # Calculate overall winner for report
    overall_scores_report = {}
    for model_key, metric in metrics.items():
        max_bpm_error = max(m['avg_bpm_error'] for m in metrics.values()) if metrics.values() else 1
        bpm_score = (1 - metric['avg_bpm_error'] / max_bpm_error) * 100 if max_bpm_error > 0 else 0
        overall_score = (bpm_score * 0.4) + (metric['key_accuracy'] * 0.3) + (metric['style_match'] * 0.3)
        overall_scores_report[model_key] = overall_score
    
    sorted_models_report = sorted(overall_scores_report.items(), key=lambda x: x[1], reverse=True)
    
    # Save detailed report
    report = {
        'test_config': {
            'num_samples': num_samples,
            'dataset': TEST_DATASET,
        },
        'models': {
            model_key: {
                'name': MODELS[model_key]['name'],
                'checkpoint': str(MODELS[model_key]['checkpoint']) if MODELS[model_key]['checkpoint'] else None,
            }
            for model_key in MODELS.keys()
        },
        'metrics': {
            model_key: {
                'avg_bpm_error': float(metrics[model_key]['avg_bpm_error']),
                'std_bpm_error': float(metrics[model_key]['std_bpm_error']),
                'min_bpm_error': float(metrics[model_key]['min_bpm_error']),
                'max_bpm_error': float(metrics[model_key]['max_bpm_error']),
                'key_accuracy': float(metrics[model_key]['key_accuracy']),
                'style_match': float(metrics[model_key]['style_match']),
                'avg_generation_time': float(metrics[model_key]['avg_generation_time']),
                'total_tests': metrics[model_key]['total_tests'],
                'successful_tests': metrics[model_key]['successful_tests'],
                'perfect_bpm_matches': metrics[model_key]['perfect_bpm_matches'],
                'perfect_key_matches': metrics[model_key]['perfect_key_matches'],
            }
            for model_key in metrics.keys()
        },
        'overall_winner': {
            'model': metrics[sorted_models_report[0][0]]['name'] if sorted_models_report else None,
            'score': float(sorted_models_report[0][1]) if sorted_models_report else None,
        },
        'test_results': [
            {
                'test_id': idx,
                'prompt': all_results['pretrained'][idx-1]['test_case']['prompt'] if idx <= len(all_results['pretrained']) else None,
                'expected_bpm': all_results['pretrained'][idx-1]['test_case']['expected_bpm'] if idx <= len(all_results['pretrained']) else None,
                'expected_key': all_results['pretrained'][idx-1]['test_case']['expected_key'] if idx <= len(all_results['pretrained']) else None,
                **{
                    f'{model_key}_bpm': all_results[model_key][idx-1]['analysis']['bpm'] if idx <= len(all_results[model_key]) and all_results[model_key][idx-1]['analysis'] else None
                    for model_key in MODELS.keys()
                },
                **{
                    f'{model_key}_key': all_results[model_key][idx-1]['analysis']['key'] if idx <= len(all_results[model_key]) and all_results[model_key][idx-1]['analysis'] else None
                    for model_key in MODELS.keys()
                },
            }
            for idx in range(1, len(test_cases) + 1)
        ]
    }

    report_file = OUTPUT_DIR / "multi_model_comparison_500_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    log_print(f"\n💾 Detailed report: {report_file}")
    log_print(f"🎵 Audio samples:")
    for model_key, model_config in MODELS.items():
        log_print(f"  {model_config['color']} {model_config['name']}: {OUTPUT_DIR / model_config['output_dir']}/")

    log_print(f"\n{'='*80}")
    log_print("✅ TEST COMPLETE")
    log_print(f"{'='*80}\n")
    log_print(f"📝 Full log saved to: {LOG_FILE}")

    log_file.close()
    return all_results

if __name__ == "__main__":
    import sys
    
    random.seed(42)
    
    start_from = 0
    num_samples = 500
    
    if len(sys.argv) > 1:
        try:
            start_from = int(sys.argv[1])
            print(f"🔄 Resuming from test {start_from}")
        except ValueError:
            print(f"⚠️  Invalid start_from value: {sys.argv[1]}, starting from beginning")
    
    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
            print(f"📊 Using {num_samples} samples")
        except ValueError:
            print(f"⚠️  Invalid num_samples value: {sys.argv[2]}, using 500")
    
    try:
        all_results = compare_all_models(start_from=start_from, num_samples=num_samples)
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

