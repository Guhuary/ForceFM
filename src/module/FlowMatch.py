from typing import Any
import copy
import torch
import torch.nn as nn
import torch.utils
from torch_scatter import scatter
from lightning.pytorch import LightningModule

from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils.pylogger import RankedLogger
from ..models.get_model import get_vector_field
log = RankedLogger(__name__, rank_zero_only=True)

class Base_FM_Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = get_vector_field(args)

        if self.args.use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema_rate)
            self.ema = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=avg_fn)
            for param in self.ema.parameters():
                param.requires_grad = False

        self.loss = nn.MSELoss(reduction='none')
        self.save_hyperparameters(
            logger=True
        )
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.debug = False

    def forward(self, data, training=False):
        """Forward pass through the model."""
        if not training and self.ema is not None:
            model = self.ema
        else:
            model = self.model
        return model(data)
    
    def loss_fn(self, tr_pred, rot_pred, tor_pred, data):
        batch = torch.cat([torch.full((n,), i) for i, n in enumerate(data.num_torsions)]).long().to(tor_pred.device)
        loss_tr = self.loss(tr_pred, data.u_tr).sum(1).mean(0)
        loss_rot = self.loss(rot_pred, data.u_rot).sum(1).mean(0)
        loss_tor = scatter(self.loss(tor_pred, data.u_tor), batch, dim=0, reduce='sum').mean(0)
        loss = loss_tr + loss_rot + loss_tor
        return loss, loss_tr, loss_rot, loss_tor

    def training_step(self, data, batch_idx):
        batch_size = data.num_graphs
        tr_pred, rot_pred, tor_pred = self(data, training=True)
        loss, tr_loss, rot_loss, tor_loss = \
					self.loss_fn(tr_pred, rot_pred, tor_pred, data=data)
        if tr_pred.isnan().any() or rot_pred.isnan().any() or tor_pred.isnan().any():
            self.debug = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            tr_loss = torch.nan_to_num(tr_loss, nan=0.0, posinf=0.0, neginf=0.0)
            rot_loss = torch.nan_to_num(rot_loss, nan=0.0, posinf=0.0, neginf=0.0)
            tor_loss = torch.nan_to_num(tor_loss, nan=0.0, posinf=0.0, neginf=0.0)
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/tr', tr_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/rot', rot_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/tor', tor_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        out_dict = {
            'loss': loss,
            'tr_loss': tr_loss.detach(),
            'rot_loss': rot_loss.detach(),
            'tor_loss': tor_loss.detach(),
        }
        return out_dict

    def validation_step(self, data, batch_idx):
        batch_size = data.num_graphs
        tr_pred, rot_pred, tor_pred = self(data, training=False)
        loss, tr_loss, rot_loss, tor_loss = \
					self.loss_fn(tr_pred, rot_pred, tor_pred, data=data)
        if tr_pred.isnan().any() or rot_pred.isnan().any() or tor_pred.isnan().any():
            self.debug = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) + 300
            tr_loss = torch.nan_to_num(tr_loss, nan=0.0, posinf=0.0, neginf=0.0) + 100
            rot_loss = torch.nan_to_num(rot_loss, nan=0.0, posinf=0.0, neginf=0.0) + 100
            tor_loss = torch.nan_to_num(tor_loss, nan=0.0, posinf=0.0, neginf=0.0) + 100
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/tr', tr_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/rot', rot_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/tor', tor_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        out_dict = {
            'loss': loss,
            'tr_loss': tr_loss.detach(),
            'rot_loss': rot_loss.detach(),
            'tor_loss': tor_loss.detach(),
        }
        return out_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        t_max = getattr(self.trainer, "max_epochs", 1000)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1,
                "name": "cosine_anneal"
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        for name, p in self.model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                log.info(
                    f"Gradients were nan for {name}, and skip_nan_grad_updates was enabled."
                    " Zeroing grad for this batch."
                )
                self.optimizer_zero_grad(epoch, batch_idx, optimizer) # type: ignore
                break
        optimizer.step(closure=optimizer_closure)
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update_parameters(self.model) # type: ignore
