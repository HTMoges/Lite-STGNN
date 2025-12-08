#!/usr/bin/env python3
"""
Configuration and Argument Parsing for Lite-STGNN
Handles command-line arguments and dataset presets.
"""
import argparse
import os
import yaml


def create_parser():
    """Create argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description='Lite-STGNN (paper-mode)')
    
    # Dataset and data configuration
    parser.add_argument('--dataset', type=str, default='electricity', 
                        choices=['electricity','traffic','exchange','weather'])
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--seq-len', type=int, default=96)
    parser.add_argument('--pred-len', type=int, default=720)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=8e-4)
    
    # Model architecture - Adjacency
    parser.add_argument('--adj-rank', type=int, default=16)
    parser.add_argument('--adj-topk', type=int, default=10)
    parser.add_argument('--adj-tau', type=float, default=1.2)
    parser.add_argument('--self-loop-alpha', type=float, default=0.2)
    
    # Model architecture - Graph propagation
    parser.add_argument('--gate-mode', type=str, default='band', choices=['flat','band'])
    parser.add_argument('--prop-orders', type=int, default=1)
    
    # Model architecture - Optional components
    parser.add_argument('--use-temporal-head', action='store_true')
    parser.add_argument('--temporal-head-ratio', type=float, default=0.5)
    parser.add_argument('--temporal-head-dropout', type=float, default=0.1)
    parser.add_argument('--use-input-residual', action='store_true')
    parser.add_argument('--input-residual-dropout', type=float, default=0.0)
    parser.add_argument('--use-feature-norm', action='store_true')
    parser.add_argument('--residual-dropout', type=float, default=0.0)
    
    # Regularization
    parser.add_argument('--lambda-sparsity', type=float, default=0.0)
    parser.add_argument('--lambda-residual', type=float, default=0.0)
    
    return parser


def get_data_root(args):
    """Get data root with environment variable override."""
    return os.environ.get('DATA_ROOT', args.data_root)


def print_config(args, model, total_params, trainable_params, data_root):
    """Print configuration summary."""
    print("=" * 80)
    print("Lite-STGNN: Lightweight Spatial-Temporal Graph Neural Network")
    print("=" * 80)
    print("\n[Basic Config]")
    print(f"  Dataset:            {args.dataset}")
    print(f"  Data Root:          {data_root}")
    print(f"  Seq Len:            {args.seq_len}")
    print(f"  Pred Len:           {args.pred_len}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Seed:               {args.seed}")
    print(f"  Learning Rate:      {args.learning_rate}")
    
    print("\n[Model Parameters]")
    print(f"  adj_rank:           {args.adj_rank}")
    print(f"  adj_topk:           {args.adj_topk}")
    print(f"  adj_tau:            {args.adj_tau}")
    print(f"  self_loop_alpha:    {args.self_loop_alpha}")
    print(f"  gate_mode:          {args.gate_mode}")
    print(f"  prop_orders:        {args.prop_orders}")
    print(f"  use_temporal_head:  {args.use_temporal_head}")
    print(f"  temporal_head_ratio: {args.temporal_head_ratio}")
    print(f"  temporal_head_dropout: {args.temporal_head_dropout}")
    print(f"  use_input_residual: {args.use_input_residual}")
    print(f"  input_residual_dropout: {args.input_residual_dropout}")
    print(f"  use_feature_norm:   {args.use_feature_norm}")
    print(f"  residual_dropout:   {args.residual_dropout}")
    print(f"  lambda_sparsity:    {args.lambda_sparsity}")
    print(f"  lambda_residual:    {args.lambda_residual}")
    
    print("\n[Run Parameters]")
    print(f"  Total Params:       {total_params}")
    print(f"  Trainable Params:   {trainable_params}")
    
    import torch
    print("\n[GPU]")
    print(f"  Use GPU:            {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:                {torch.cuda.current_device()}")
        print(f"  GPU Name:           {torch.cuda.get_device_name(0)}")
    
    print("=" * 80)
    print(f">>>>>>>start training: {args.dataset}_L{args.seq_len}_P{args.pred_len}_seed{args.seed}>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(f"Parameters: {total_params}")
    print("=" * 80)
