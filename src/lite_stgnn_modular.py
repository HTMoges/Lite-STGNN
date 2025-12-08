#!/usr/bin/env python3
"""
Lite-STGNN Main Training Script (Modularized)
Clean modular architecture for paper reproduction.
"""
import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from datasets import create_dataloaders
from model import LiteSTGNN
from trainer import LiteSTGNN_Forecaster
from config import create_parser, get_data_root, print_config


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    data_root = get_data_root(args)
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = create_dataloaders(
        dataset_name=args.dataset,
        data_root=data_root,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count() or 2)
    )
    
    sample_x, _ = train_ds[0]
    n_features = sample_x.shape[-1]

    model_arch = LiteSTGNN(
        n_nodes=n_features,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        adj_rank=args.adj_rank,
        adj_topk=args.adj_topk,
        adj_tau=args.adj_tau,
        self_loop_alpha=args.self_loop_alpha,
        gate_mode=args.gate_mode,
        prop_orders=args.prop_orders,
        use_temporal_head=args.use_temporal_head,
        temporal_head_ratio=args.temporal_head_ratio,
        temporal_head_dropout=args.temporal_head_dropout,
        use_input_residual=args.use_input_residual,
        input_residual_dropout=args.input_residual_dropout,
        use_feature_norm=args.use_feature_norm,
        residual_dropout=args.residual_dropout,
    )

    model = LiteSTGNN_Forecaster(
        model=model_arch,
        learning_rate=args.learning_rate,
        lambda_sparsity=args.lambda_sparsity,
        lambda_residual=args.lambda_residual,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print_config(args, model, total_params, trainable_params, data_root)

    early_stop = EarlyStopping(monitor='val_mse_mean_horizons', patience=10, mode='min', verbose=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='32',
        logger=None,
        enable_progress_bar=False,
        callbacks=[early_stop],
        enable_model_summary=False,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_minutes = (time.time() - t0) / 60.0
    
    print("=" * 80)
    print(f"Training complete in {train_minutes:.2f} minutes")
    print("=" * 80)
    
    print(f"\n>>>>>>>testing: {args.dataset}_L{args.seq_len}_P{args.pred_len}_seed{args.seed}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
