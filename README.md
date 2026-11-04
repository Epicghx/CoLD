# CoLD: Cross-view Lead Decoupling

Code for *Cross-View Lead Decoupling for Self-Supervised ECG Pretraining on Rare Diagnostic Statements*.

CoLD is a plug-and-play augmentation module for joint-embedding pretraining on 12-lead ECGs. It draws a gain and an offset for every lead of each view from one Gaussian copula, so the two views of a recording differ lead by lead. The lead coefficient `lam` sets how much the gains and offsets vary from lead to lead, and the view coefficient `rho` sets how much of this variation the two views share. The default is `lam = rho = 0`.

## Setup

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh    # all stages on random signals, about a minute
```

## Data

```bash
python prepare_ptbxl.py --src /path/to/ptb-xl --dst data/ptbxl      # PTB-XL 1.0.3
python prepare_code15.py --src /path/to/code15 --dst data/code15    # CODE-15%, one recording per patient
```

## Pretraining

```bash
python pretrain.py --config configs/pretrain_cold_vit.yaml
```

| config | objective | augmentation | encoder |
|---|---|---|---|
| `pretrain_cold_vit.yaml` | MoCo-v2 | CoLD | ViT-Tiny |
| `pretrain_moco_vit.yaml` | MoCo-v2 | standard (one global gain) | ViT-Tiny |
| `pretrain_control_vit.yaml` | MoCo-v2 | crop, shift and noise only | ViT-Tiny |
| `pretrain_cold_resnet18.yaml` | MoCo-v2 | CoLD | ResNet-1D-18 |
| `pretrain_cold_barlow_vit.yaml` | Barlow Twins | CoLD | ViT-Tiny |
| `pretrain_cold_vicreg_vit.yaml` | VICReg | CoLD | ViT-Tiny |

The encoder is saved to `runs/<name>/encoder.pt`.

## Spread matching

```bash
python spread_matching.py --ptbxl data/ptbxl
```

On PTB-XL this requires a gain ratio of at least 3.53 and an offset SD of at least 0.29. The defaults, gains in [0.5, 2] and `sigma_b = 0.5`, meet both bounds.

## Fine-tuning

```bash
python finetune.py --config configs/finetune_ptbxl.yaml --ckpt runs/cold_vit/encoder.pt --seed 42
python finetune.py --config configs/finetune_ptbxl.yaml --ckpt none    # ViT-Tiny from scratch
```

Test metrics, including tail-, mid- and head-AUPRC, are written to `runs/.../test_metrics.json`. Set `data.label_set` and `data.train_frac` in the config for other PTB-XL label sets and label fractions.

## Using CoLD in other code

```python
from cold.views import TwoViews

views = TwoViews(amp="cold", s_min=0.5, s_max=2.0, sigma_b=0.5, lam=0.0, rho=0.0)
v1, v2 = views(x)    # x: (12, T) tensor, standardised per lead
```

## License

MIT
