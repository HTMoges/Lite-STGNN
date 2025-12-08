#!/usr/bin/env python3
"""
Training and Evaluation Module for Lite-STGNN
PyTorch Lightning training loop with multi-horizon evaluation.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional


class LiteSTGNN_Forecaster(pl.LightningModule):
    def __init__(self, model, learning_rate: float = 8e-4, milestones: Optional[list] = None,
                 lambda_sparsity: float = 0.0, lambda_residual: float = 0.0):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.milestones = milestones or [30, 60, 90]
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_residual = float(lambda_residual)
        
        self._reset_train_acc()
        self._reset_val_acc()

    def _get_eval_horizons(self) -> list:
        L = self.model.pred_len
        canonical = [96, 192, 336, 720]
        hs = [h for h in canonical if h <= L]
        if len(hs) == 0:
            q1, q2, q3 = max(1, L // 4), max(1, L // 2), max(1, (3 * L) // 4)
            hs = sorted({q1, q2, q3, L})
        elif hs[-1] != L:
            hs = hs + ([L] if L not in hs else [])
        return hs[:4] if len(hs) >= 4 else hs

    def _reset_train_acc(self):
        self.train_loss_sum = 0.0
        self.train_count = 0

    def _reset_val_acc(self):
        hs = self._get_eval_horizons()
        self.val_abs = {h: 0.0 for h in hs}
        self.val_sq = {h: 0.0 for h in hs}
        self.val_cnt = {h: 0 for h in hs}
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        if self.lambda_sparsity > 0.0 and hasattr(self.model, 'adj'):
            A = self.model.adj()
            sparsity = A.sum() / A.numel()
            loss = loss + self.lambda_sparsity * sparsity
        
        if self.lambda_residual > 0.0:
            y_lin = self.model.backbone(self.model.pre_norm(x))
            res_mag = (y_hat - y_lin).pow(2).mean()
            loss = loss + self.lambda_residual * res_mag
        
        self.train_loss_sum += float(loss.detach().cpu())
        self.train_count += 1
        self.log('train_mse', loss, prog_bar=True)
        return loss

    def _eval(self, batch, prefix: str):
        x, y = batch
        y_hat = self(x)
        
        hs = self._get_eval_horizons()
        for h in hs:
            y_h = y[:, :h, :]
            yhat_h = y_hat[:, :h, :]
            abs_err = torch.abs(yhat_h - y_h).sum().item()
            sq_err = ((yhat_h - y_h) ** 2).sum().item()
            cnt = y_h.numel()
            self.val_abs[h] += abs_err
            self.val_sq[h] += sq_err
            self.val_cnt[h] += cnt
        
        mse_full = F.mse_loss(y_hat, y)
        mae_full = F.l1_loss(y_hat, y)
        self.log(f'{prefix}_mse', mse_full, prog_bar=(prefix=='val'))
        self.log(f'{prefix}_mae', mae_full)
        return mse_full

    def validation_step(self, batch, batch_idx):
        return self._eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._eval(batch, 'test')

    def on_validation_epoch_start(self):
        self._reset_val_acc()

    def on_train_epoch_start(self):
        self._reset_train_acc()

    def on_validation_epoch_end(self):
        if getattr(self.trainer, 'sanity_checking', False):
            return
        
        hs = self._get_eval_horizons()
        mae_h = {}
        mse_h = {}
        for h in hs:
            cnt = max(1, self.val_cnt[h])
            mae_h[h] = self.val_abs[h] / cnt
            mse_h[h] = self.val_sq[h] / cnt
        
        ordered = hs[:4]
        mae_mean_horizons = sum(mae_h[h] for h in ordered) / len(ordered)
        mse_mean_horizons = sum(mse_h[h] for h in ordered) / len(ordered)
        
        # Log silently (no console output from callback)
        self.log('val_mse_mean_horizons', torch.tensor(mse_mean_horizons, device=self.device), prog_bar=False, logger=False)
        self.log('val_mae_mean_horizons', torch.tensor(mae_mean_horizons, device=self.device), logger=False)
        
        for h in ordered:
            self.log(f'val_mae_at_{h}', torch.tensor(mae_h[h], device=self.device), logger=False)
            self.log(f'val_mse_at_{h}', torch.tensor(mse_h[h], device=self.device), logger=False)
        
        epoch = self.current_epoch
        train_mse_epoch = (self.train_loss_sum / max(1, self.train_count)) if self.train_count else float('nan')
        
        improvement_indicator = ""
        if mse_mean_horizons < self.best_val_loss:
            self.best_val_loss = mse_mean_horizons
            improvement_indicator = " ✓"
        
        print(f"Epoch {epoch:02d}: Train={train_mse_epoch:.6f} Val={mse_mean_horizons:.6f}{improvement_indicator}")

    def on_test_epoch_start(self):
        self._reset_val_acc()

    def on_test_epoch_end(self):
        hs = self._get_eval_horizons()
        mae_h = {}
        mse_h = {}
        for h in hs:
            cnt = max(1, self.val_cnt[h])
            mae_h[h] = self.val_abs[h] / cnt
            mse_h[h] = self.val_sq[h] / cnt
        
        ordered = hs[:4]
        mae_mean_horizons = sum(mae_h[h] for h in ordered) / len(ordered)
        mse_mean_horizons = sum(mse_h[h] for h in ordered) / len(ordered)

        # Log silently for script extraction
        self.log('test_mae_mean_horizons', torch.tensor(mae_mean_horizons, device=self.device), logger=False)
        self.log('test_mse_mean_horizons', torch.tensor(mse_mean_horizons, device=self.device), logger=False)
        
        for h in ordered:
            self.log(f'test_mae_at_{h}', torch.tensor(mae_h[h], device=self.device), logger=False)
            self.log(f'test_mse_at_{h}', torch.tensor(mse_h[h], device=self.device), logger=False)

        # Clean, compact test output
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"MSE: ", end="")
        for h in ordered:
            print(f"H{h}={mse_h[h]:.4f}  ", end="")
        print(f"Mean={mse_mean_horizons:.4f}")
        print(f"MAE: ", end="")
        for h in ordered:
            print(f"H{h}={mae_h[h]:.4f}  ", end="")
        print(f"Mean={mae_mean_horizons:.4f}")
        print(f"{'='*80}")
        print(f"test_mae_mean_horizons {mae_mean_horizons}")
        print(f"test_mse_mean_horizons {mse_mean_horizons}")
        print(f"{'='*80}\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
