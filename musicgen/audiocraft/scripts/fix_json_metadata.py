#!/usr/bin/env python3
"""Fix JSON metadata files to match MusicInfo dataclass fields"""

import json
from pathlib import Path

def fix_metadata_files(dataset_dir: Path):
    """Update metadata to match MusicInfo dataclass fields"""
    json_files = list(dataset_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON metadata files")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create correct fields
        new_data = {}
        
        # Required MusicInfo fields (all optional in dataclass but needed for training)
        new_data['title'] = data.get('title', json_file.stem)
        new_data['artist'] = data.get('artist', 'Beatworld Collective')
        new_data['key'] = data.get('key', '')
        new_data['bpm'] = float(data.get('bpm', 0)) if data.get('bpm') else None
        new_data['genre'] = data.get('genre', data.get('style', ''))
        new_data['description'] = data.get('description', 'instrumental music')
        new_data['name'] = data.get('name', data.get('title', json_file.stem))
        new_data['instrument'] = data.get('instrumentation', data.get('instrument', ''))
        
        # moods is a list
        mood_val = data.get('mood', '')
        if mood_val:
            new_data['moods'] = [mood_val] if isinstance(mood_val, str) else mood_val
        else:
            new_data['moods'] = []
        
        # keywords is optional
        new_data['keywords'] = data.get('keywords', [])
        
        # Remove old incompatible fields
        # Remove: album, mood (singular), instrumentation
        
        # Write back
        with open(json_file, 'w') as f:
            json.dump(new_data, f, indent=2)
    
    print(f"✅ Fixed {len(json_files)} JSON metadata files")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_json_metadata.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    fix_metadata_files(dataset_dir)



