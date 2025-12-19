#!/bin/bash
# MusicGen STEREO-MELODY-LARGE with LoRA
# Memory-efficient training: ~12-15 GB instead of ~51 GB
# Only trains 20-50M adapter parameters instead of 2.5B full parameters

set -e

echo "======================================================================"
echo "  MusicGen STEREO-MELODY-LARGE - LoRA Training"
echo "======================================================================"

# Check GPU
echo "Step 1: Checking GPU..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU Memory Usage: ${GPU_MEM}MB"

if [ "$GPU_MEM" -gt 5000 ]; then
    echo "Clearing GPU memory..."
    pkill -9 -f audiocraft.train || true
    pkill -9 -f train_large_lora || true
    sleep 3
fi

# Activate environment
source /home/ubuntu/beats/venv/bin/activate
cd /home/ubuntu/beats/musicgen/training/audiocraft

echo ""
echo "Step 2: Starting LoRA training..."
echo "Model: facebook/musicgen-stereo-melody-large (2.5B params)"
echo ""
echo "LoRA SETTINGS:"
echo "  ✓ LARGE stereo-melody model (2.5B params)"
echo "  ✓ Only training adapters (~20-50M params)"
echo "  ✓ Memory: ~12-15 GB (vs ~51 GB full fine-tuning)"
echo "  ✓ Original dataset (1,167 samples)"
echo "  ✓ LoRA Rank: 16 (adapter capacity)"
echo "  ✓ Learning Rate: 1e-4 (higher for LoRA)"
echo "  ✓ Batch size: 2"
echo "  ✓ Epochs: 5"
echo "  ✓ Duration: 20s per sample"
echo "======================================================================"
echo ""

# Run LoRA training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_large_lora.py \
    --model_name facebook/musicgen-stereo-melody-large \
    --dataset_path /home/ubuntu/beats/musicgen/training/audiocraft/egs/beats_full_complete/train/normalized_audio \
    --output_dir /home/ubuntu/beats/musicgen/training/lora_checkpoints \
    --epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --duration 20 \
    --save_every 500 \
    2>&1 | tee /home/ubuntu/beats/musicgen/training/training_large_lora.log

echo ""
echo "======================================================================"
echo "Training completed!"
echo "Log: /home/ubuntu/beats/musicgen/training/training_large_lora.log"
echo "LoRA Adapters: /home/ubuntu/beats/musicgen/training/lora_checkpoints/"
echo ""
echo "LoRA BENEFITS:"
echo "  ✓ 90-95% of full fine-tuning quality"
echo "  ✓ 3-4x less memory (fits in GPU!)"
echo "  ✓ Faster training (fewer params to update)"
echo "  ✓ Less overfitting risk"
echo "  ✓ Can train multiple adapters for different styles"
echo "======================================================================"
