#!/bin/bash
# Safe cleanup script - ONLY deletes checkpoints from synthetic data training
# This script is very careful to only delete from checkpoints_synthetic directory

set -e

SYNTHETIC_CHECKPOINT_DIR="/home/ubuntu/beats/musicgen/training/checkpoints_synthetic"

echo "======================================================================"
echo "  Cleanup Script - Synthetic Data Checkpoints Only"
echo "======================================================================"
echo ""
echo "⚠️  WARNING: This will delete ALL files in:"
echo "   $SYNTHETIC_CHECKPOINT_DIR"
echo ""
echo "This directory contains checkpoints from:"
echo "  ✓ Full fine-tuning with synthetic data"
echo "  ✓ Training script: train_synthetic_full.py"
echo ""
echo "Other checkpoint directories will NOT be touched:"
echo "  ✓ /home/ubuntu/beats/musicgen/training/checkpoints/ (preserved)"
echo "  ✓ /home/ubuntu/beats/musicgen/training/checkpoints_simple/ (preserved)"
echo "  ✓ /home/ubuntu/beats/musicgen/training/lora_checkpoints_synthetic/ (preserved)"
echo ""

# Check if directory exists
if [ ! -d "$SYNTHETIC_CHECKPOINT_DIR" ]; then
    echo "❌ Directory does not exist: $SYNTHETIC_CHECKPOINT_DIR"
    exit 1
fi

# Count files and directories
FILE_COUNT=$(find "$SYNTHETIC_CHECKPOINT_DIR" -mindepth 1 -type f | wc -l)
DIR_COUNT=$(find "$SYNTHETIC_CHECKPOINT_DIR" -mindepth 1 -type d | wc -l)

echo "Found in $SYNTHETIC_CHECKPOINT_DIR:"
echo "  Files: $FILE_COUNT"
echo "  Directories: $DIR_COUNT"
echo ""

if [ "$FILE_COUNT" -eq 0 ] && [ "$DIR_COUNT" -eq 0 ]; then
    echo "✅ Directory is already empty. Nothing to delete."
    exit 0
fi

# Show what will be deleted
echo "Files/directories that will be deleted:"
find "$SYNTHETIC_CHECKPOINT_DIR" -mindepth 1 -maxdepth 2 | head -20
if [ "$FILE_COUNT" -gt 20 ] || [ "$DIR_COUNT" -gt 20 ]; then
    echo "... (and more)"
fi
echo ""

# Ask for confirmation
read -p "Are you sure you want to delete these synthetic data checkpoints? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled. No files deleted."
    exit 0
fi

# Delete only from synthetic checkpoint directory
echo ""
echo "🗑️  Deleting synthetic data checkpoints..."
rm -rf "$SYNTHETIC_CHECKPOINT_DIR"/* "$SYNTHETIC_CHECKPOINT_DIR"/.[!.]* 2>/dev/null || true

# Verify deletion
REMAINING=$(find "$SYNTHETIC_CHECKPOINT_DIR" -mindepth 1 | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✅ Successfully deleted all synthetic data checkpoints."
    echo "   Directory is now empty: $SYNTHETIC_CHECKPOINT_DIR"
else
    echo "⚠️  Warning: $REMAINING items still remain in directory."
fi

echo ""
echo "======================================================================"
echo "Cleanup complete!"
echo "======================================================================"

