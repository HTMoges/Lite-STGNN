#!/bin/bash
# Reproduce paper results for Lite-STGNN

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
EPOCHS=50
SEED=42

echo "Running Lite-STGNN experiments..."
echo "Data root: ${DATA_ROOT}"
echo ""

# Electricity
echo "Training Electricity..."
python src/lite_stgnn_modular.py \
    --dataset electricity \
    --data-root ${DATA_ROOT} \
    --seq-len 96 --pred-len 720 --epochs ${EPOCHS} --seed ${SEED} \
    --learning-rate 5e-4 \
    --adj-rank 16 --adj-topk 10 --adj-tau 1.2 \
    --use-input-residual \
    --use-temporal-head --temporal-head-ratio 0.5 \
    --residual-dropout 0.1 \
    --gate-mode band --prop-orders 2

# Traffic
echo "Training Traffic..."
python src/lite_stgnn_modular.py \
    --dataset traffic \
    --data-root ${DATA_ROOT} \
    --seq-len 96 --pred-len 720 --epochs ${EPOCHS} --seed ${SEED} \
    --learning-rate 8e-4 \
    --adj-rank 24 --adj-topk 10 --adj-tau 1.0 \
    --use-input-residual \
    --use-temporal-head --temporal-head-ratio 0.7 \
    --residual-dropout 0.1 \
    --gate-mode band --prop-orders 3

# Exchange
echo "Training Exchange..."
python src/lite_stgnn_modular.py \
    --dataset exchange \
    --data-root ${DATA_ROOT} \
    --seq-len 96 --pred-len 720 --epochs ${EPOCHS} --seed ${SEED} \
    --learning-rate 1e-3 \
    --adj-rank 4 --adj-topk 4 --adj-tau 1.0 \
    --use-input-residual \
    --gate-mode band --prop-orders 1

# Weather
echo "Training Weather..."
python src/lite_stgnn_modular.py \
    --dataset weather \
    --data-root ${DATA_ROOT} \
    --seq-len 96 --pred-len 720 --epochs ${EPOCHS} --seed ${SEED} \
    --learning-rate 5e-4 \
    --adj-rank 16 --adj-topk 16 --adj-tau 0.8 \
    --use-input-residual \
    --use-temporal-head --temporal-head-ratio 0.3 \
    --gate-mode band --prop-orders 2

echo "All experiments completed."