#!/usr/bin/env python3
"""Add ALL fields from MusicInfo dataclass to JSON files"""

import json
from pathlib import Path

def add_all_music_info_fields(dataset_dir: Path):
    """Add all MusicInfo fields to JSON files"""
    json_files = list(dataset_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON metadata files")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # ALL MusicInfo fields (from dataclass)
        new_data = {}
        new_data['title'] = data.get('title', json_file.stem)
        new_data['artist'] = data.get('artist', 'Beatworld Collective')
        new_data['key'] = data.get('key', '')
        new_data['bpm'] = data.get('bpm')  # Can be None
        new_data['genre'] = data.get('genre', '')
        new_data['moods'] = data.get('moods', [])
        new_data['keywords'] = data.get('keywords', [])
        new_data['description'] = data.get('description', 'instrumental music')
        new_data['name'] = data.get('name', data.get('title', json_file.stem))
        new_data['instrument'] = data.get('instrument', '')
        
        # Write back
        with open(json_file, 'w') as f:
            json.dump(new_data, f, indent=2)
    
    print(f"✅ Updated {len(json_files)} JSON metadata files with all MusicInfo fields")


if __name__ == '__main__':
    import sys
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    add_all_music_info_fields(dataset_dir)



