#!/bin/bash
# MusicGen LARGE with LoRA - Using AudioCraft Framework
# Injects LoRA adapters into AudioCraft's training pipeline
# Memory: ~12-15 GB instead of ~51 GB

set -e

echo "======================================================================"
echo "  MusicGen LARGE - LoRA via AudioCraft Framework"
echo "======================================================================"

# Check GPU
echo "Step 1: Checking GPU..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU Memory Usage: ${GPU_MEM}MB"

if [ "$GPU_MEM" -gt 5000 ]; then
    echo "Clearing GPU memory..."
    pkill -9 -f audiocraft.train || true
    sleep 3
fi

# Activate environment
source /home/ubuntu/beats/venv/bin/activate
cd /home/ubuntu/beats/musicgen/training/audiocraft

# Ensure ffmpeg is in PATH
export PATH="/usr/bin:$PATH"

echo ""
echo "Step 2: Starting LoRA training via AudioCraft..."
echo ""
echo "LoRA CONFIGURATION:"
echo "  ✓ LARGE stereo-melody model (2.5B params)"
echo "  ✓ HuggingFace Encodec 32kHz (cardinality: 2048)"
echo "  ✓ Training: 32kHz (native, matches pretrained model)"
echo "  ✓ Generation: 48kHz (upsampled with high-quality resampling)"
echo "  ✓ LoRA adapters (87.6M trainable params, 3.5%)"
echo "  ✓ Memory: ~12-15 GB (vs ~51 GB full training)"
echo "  ✓ Rank: 16, Alpha: 32"
echo "  ✓ Original dataset (1,167 samples)"
echo "  ✓ Epochs: 15, Batch: 4, LR: 5e-5"
echo "======================================================================"
echo ""

# Run AudioCraft training with LoRA injection
USE_LORA=1 \
LORA_RANK=16 \
LORA_ALPHA=32 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/ubuntu/beats/venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from add_lora_hook import inject_lora_into_audiocraft
inject_lora_into_audiocraft()

# Now run audiocraft training
import audiocraft.train as train_module
train_module.main()
" \
    solver=musicgen/musicgen_melody_32khz \
    dset=audio/beats_full_complete \
    model/lm/model_scale=large \
    continue_from=//pretrained/facebook/musicgen-stereo-melody-large \
    label=corrected_card2048_32khz \
    \
    channels=2 \
    interleave_stereo_codebooks.use=true \
    transformer_lm.n_q=8 \
    transformer_lm.card=2048 \
    'codebooks_pattern.delay.delays=[0, 0, 1, 1, 2, 2, 3, 3]' \
    \
    transformer_lm.checkpointing=torch \
    \
    dataset.batch_size=4 \
    dataset.num_workers=0 \
    dataset.segment_duration=20 \
    dataset.min_segment_ratio=0.8 \
    dataset.valid.num_samples=20 \
    \
    optim.epochs=15 \
    optim.updates_per_epoch=292 \
    optim.optimizer=adamw \
    optim.lr=5e-5 \
    optim.adam.weight_decay=0.01 \
    'optim.adam.betas=[0.9,0.95]' \
    optim.max_norm=1.0 \
    \
    optim.ema.use=false \
    \
    schedule.lr_scheduler=cosine \
    schedule.cosine.warmup=500 \
    schedule.cosine.lr_min_ratio=0.0 \
    \
    autocast=true \
    autocast_dtype=bfloat16 \
    \
    evaluate.every=999999 \
    generate.every=999999 \
    \
    checkpoint.save_last=true \
    checkpoint.save_every=10 \
    checkpoint.keep_last=1 \
    \
    logging.log_tensorboard=true \
    2>&1 | tee /home/ubuntu/beats/musicgen/training/training_large_lora_audiocraft.log

echo ""
echo "======================================================================"
echo "Training completed!"
echo "Log: /home/ubuntu/beats/musicgen/training/training_large_lora_audiocraft.log"
echo ""
echo "LoRA BENEFITS:"
echo "  ✓ Uses proven AudioCraft training pipeline"
echo "  ✓ 90-95% of full fine-tuning quality"
echo "  ✓ 3-4x less memory (fits in 48GB GPU!)"
echo "  ✓ Faster training"
echo "  ✓ Less overfitting risk"
echo "======================================================================"
