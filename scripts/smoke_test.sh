#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python pretrain.py --config configs/pretrain_cold_vit.yaml --synthetic 64 --epochs 1 --batch_size 16 --out runs/smoke
python finetune.py --config configs/finetune_ptbxl.yaml --ckpt runs/smoke/cold_vit/encoder.pt --synthetic 64 --epochs 1 --batch_size 16 --out runs/smoke
for c in moco_vit control_vit cold_resnet18 cold_barlow_vit cold_vicreg_vit; do
  python pretrain.py --config configs/pretrain_$c.yaml --synthetic 32 --epochs 1 --batch_size 16 --out runs/smoke
done
