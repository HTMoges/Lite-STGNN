# Lite-STGNN

Official PyTorch implementation of **Lite-STGNN: A Lightweight Spatial-Temporal Graph Neural Network for Long-Term Forecasting**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Get Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
All datasets can be obtained from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

Place datasets in `./data/`:
```
data/
├── electricity.csv
├── traffic.csv
├── exchange_rate.csv
└── weather.csv
```

### 3. Run Experiments

Using configuration files:
```bash
python src/lite_stgnn_modular.py --config configs/electricity.yaml
python src/lite_stgnn_modular.py --config configs/traffic.yaml
python src/lite_stgnn_modular.py --config configs/exchange.yaml
python src/lite_stgnn_modular.py --config configs/weather.yaml
```

Or with command-line arguments:
```bash
python src/lite_stgnn_modular.py \
  --dataset electricity \
  --data-root ./data \
  --seq-len 96 --pred-len 720 --epochs 50 \
  --learning-rate 5e-4 \
  --adj-rank 16 --adj-topk 10 --adj-tau 1.2 \
  --use-input-residual \
  --use-temporal-head --temporal-head-ratio 0.5 \
  --gate-mode band --prop-orders 2
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{litestgnn2024,
  title={Lite-STGNN: A Lightweight Spatial-Temporal Graph Neural Network for Long-Term Forecasting},
  author={H.T. Moges and D. Moodley},
  booktitle={ICAART},
  year={2025}
}
```

## Acknowledgement

We appreciate the following repositories for their valuable code and datasets:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

## License

MIT License
