#!/bin/bash
# Quick test training with CPU (100 steps only)
# This is just to verify the entire pipeline works correctly

echo "🚀 Starting quick test training on CPU..."
echo "This will train for 100 steps to verify the pipeline"
echo ""

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  uv run python scripts/train_pytorch.py pi0_custom_robot \
  --exp-name=grab_bottle_test_cpu \
  --overwrite \
  --num-train-steps=100 \
  --log-interval=10 \
  --save-interval=50

echo ""
echo "✅ Test completed! Check the output above for any errors."
echo "If successful, you can proceed with full GPU training."
