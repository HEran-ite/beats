#!/usr/bin/env python3
"""Update JSON metadata files to include all required fields for MusicDataset"""

import json
from pathlib import Path

def update_metadata_files(dataset_dir: Path):
    """Add missing fields to all JSON metadata files"""
    json_files = list(dataset_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON metadata files")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Add missing required fields if they don't exist
        if 'artist' not in data:
            data['artist'] = data.get('title', 'Unknown').split(' - ')[0] if ' - ' in data.get('title', '') else 'Beatworld Collective'
        
        if 'album' not in data:
            data['album'] = 'Beats Collection'
        
        if 'title' not in data:
            data['title'] = json_file.stem
        
        # Ensure all fields have values (not None)
        data['key'] = data.get('key', '') or ''
        data['bpm'] = data.get('bpm') if data.get('bpm') is not None else 0.0
        data['genre'] = data.get('genre', '') or ''
        data['mood'] = data.get('mood', '') or ''
        data['instrumentation'] = data.get('instrumentation', '') or ''
        data['description'] = data.get('description', 'instrumental music')
        
        # Write back
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"✅ Updated {len(json_files)} JSON metadata files")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python update_json_metadata.py <dataset_dir>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    update_metadata_files(dataset_dir)



