# CoLD: Cross-view Lead Decoupling

Code for *Cross-View Lead Decoupling for Self-Supervised ECG Pretraining on Rare Diagnostic Statements*.

CoLD is a view generator for two-view self-supervised pretraining on 12-lead ECGs. It draws a gain and an offset for every lead of each view from one Gaussian copula, so the two views of a recording differ lead by lead. The lead coefficient `lam` sets how strongly the draws are correlated across leads, and the view coefficient `rho` sets how strongly they are correlated between the two views. The default `lam = rho = 0` draws every gain and offset independently.

## Setup

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh          # every stage on random signals, about a minute
```

## Data

```bash
# PTB-XL 1.0.3 from PhysioNet -> data/ptbxl/{X100.npy, ptbxl_database.csv, scp_statements.csv}
python prepare_ptbxl.py --src /path/to/ptb-xl --dst data/ptbxl

# CODE-15% from Zenodo -> data/code15/<exam_id>.npy, one exam per patient
python prepare_code15.py --src /path/to/code15 --dst data/code15
```

Any other corpus works as a folder of `(12, 1000)` arrays at 100 Hz, one `.npy` per recording.

## Pretraining

```bash
python pretrain.py --config configs/pretrain_cold_vit.yaml
```

| config | objective | views | encoder |
|---|---|---|---|
| `pretrain_cold_vit.yaml` | MoCo-v2 | CoLD | ViT-Tiny |
| `pretrain_moco_vit.yaml` | MoCo-v2 | one global gain per view | ViT-Tiny |
| `pretrain_control_vit.yaml` | MoCo-v2 | crop, shift and noise only | ViT-Tiny |
| `pretrain_cold_resnet18.yaml` | MoCo-v2 | CoLD | ResNet-1D-18 |
| `pretrain_cold_barlow_vit.yaml` | Barlow Twins | CoLD | ViT-Tiny |
| `pretrain_cold_vicreg_vit.yaml` | VICReg | CoLD | ViT-Tiny |

SimCLR and BYOL are also in `cold/models/objectives.py` (`objective: simclr` or `byol`). The encoder is saved to `runs/<name>/encoder.pt`.

## Spread matching

```bash
python spread_matching.py --ptbxl data/ptbxl
```

On PTB-XL this gives `delta_s = 0.492`, `delta_b = 0.393`, a gain ratio of at least 3.53 and an offset SD of at least 0.29. The defaults, gains in [0.5, 2] and `sigma_b = 0.5`, meet both bounds.

## Fine-tuning

```bash
python finetune.py --config configs/finetune_ptbxl.yaml --ckpt runs/cold_vit/encoder.pt --seed 42
python finetune.py --config configs/finetune_ptbxl.yaml --ckpt none        # ViT-Tiny from scratch
```

The script keeps the epoch with the best validation macro-AUROC (fold 9) and writes test metrics on fold 10, including head-, mid- and tail-AUPRC by training prevalence (>= 5%, 1-5%, < 1%), to `runs/.../test_metrics.json`. Set `data.label_set` for the PTB-XL label sets and `data.train_frac` for fewer labels.

## Using the view generator elsewhere

```python
from cold.views import TwoViews

views = TwoViews(amp="cold", s_min=0.5, s_max=2.0, sigma_b=0.5, lam=0.0, rho=0.0)
v1, v2 = views(x)        # x: (12, T) tensor, standardised per lead
```

## License

MIT
