#!/usr/bin/env python3
"""
Validate synthetic data format for MusicGen training
Checks JSONL format, audio file existence, and data integrity
"""

import json
import sys
from pathlib import Path
import torchaudio

def validate_jsonl(jsonl_path: Path):
    """Validate JSONL file format and content"""
    print(f"\n{'='*70}")
    print(f"Validating JSONL: {jsonl_path}")
    print(f"{'='*70}\n")
    
    if not jsonl_path.exists():
        print(f"❌ JSONL file not found: {jsonl_path}")
        return False
    
    required_fields = ['path']
    optional_fields = ['description', 'duration', 'sample_rate', 'meta']
    
    valid_samples = 0
    invalid_samples = 0
    missing_files = 0
    audio_errors = 0
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Check required fields
                missing_fields = [field for field in required_fields if field not in entry]
                if missing_fields:
                    print(f"⚠️  Line {line_num}: Missing required fields: {missing_fields}")
                    invalid_samples += 1
                    continue
                
                # Check audio file exists
                audio_path = Path(entry['path'])
                if not audio_path.exists():
                    print(f"⚠️  Line {line_num}: Audio file not found: {audio_path}")
                    missing_files += 1
                    invalid_samples += 1
                    continue
                
                # Try to load audio file
                try:
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                    
                    # Validate audio properties
                    if info.num_channels not in [1, 2]:
                        print(f"⚠️  Line {line_num}: Unexpected channels ({info.num_channels}): {audio_path}")
                        audio_errors += 1
                    
                    # Update entry with actual duration if missing
                    if 'duration' not in entry or entry['duration'] is None:
                        entry['duration'] = duration
                    
                except Exception as e:
                    print(f"⚠️  Line {line_num}: Error loading audio {audio_path}: {e}")
                    audio_errors += 1
                    invalid_samples += 1
                    continue
                
                valid_samples += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: Invalid JSON: {e}")
                invalid_samples += 1
            except Exception as e:
                print(f"❌ Line {line_num}: Error: {e}")
                invalid_samples += 1
    
    # Summary
    total = valid_samples + invalid_samples
    print(f"\n{'='*70}")
    print(f"Validation Summary:")
    print(f"  Total entries: {total}")
    print(f"  Valid entries: {valid_samples}")
    print(f"  Invalid entries: {invalid_samples}")
    print(f"  Missing files: {missing_files}")
    print(f"  Audio errors: {audio_errors}")
    print(f"{'='*70}\n")
    
    if valid_samples == 0:
        print("❌ No valid samples found!")
        return False
    
    if invalid_samples > 0:
        print(f"⚠️  Warning: {invalid_samples} invalid entries found, but {valid_samples} valid entries available")
    
    print(f"✅ Validation complete: {valid_samples} valid samples ready for training")
    return True

def main():
    dataset_path = Path("/home/ubuntu/beats/musicgen/training/audiocraft/egs/synthetic_audio/train")
    jsonl_path = dataset_path / "data.jsonl"
    
    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    
    success = validate_jsonl(jsonl_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

