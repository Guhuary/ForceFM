from collections import OrderedDict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
import torch.utils
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_scatter import scatter

from ..models.get_model import get_vector_field
from ..utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _has_basefm_segment(key: str) -> bool:
    return "basefm" in key.split(".")


def _resolve_path(path_like) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    try:
        original_cwd = Path(hydra.utils.get_original_cwd())
    except ValueError:
        return path
    return original_cwd / path


def _load_model_args_from_hparams(hparams_path: Path):
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Base model hparams.yaml not found: {hparams_path}")

    conf = OmegaConf.load(hparams_path)
    if "args" in conf:
        model_args = conf.args
    elif "model" in conf and "args" in conf.model:
        model_args = conf.model.args
    else:
        raise KeyError(f"Could not find model args in {hparams_path}")

    return OmegaConf.create(OmegaConf.to_container(model_args, resolve=True))


def load_base_vector_field(args):
    basemodel = getattr(args, "basemodel", None)
    if basemodel is None:
        raise ValueError("cfg.model.args.basemodel must point to the frozen base model weights.")

    checkpoint_path = _resolve_path(basemodel)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base model checkpoint not found: {checkpoint_path}")

    hparams_path = checkpoint_path.parent / "hparams.yaml"
    base_args = _load_model_args_from_hparams(hparams_path)
    base_model = get_vector_field(base_args)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Base model file must be a plain model state_dict: {checkpoint_path}")
    base_model.load_state_dict(state_dict, strict=True)
    log.info(f"Loaded base vector field state_dict from {checkpoint_path}.")
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    return base_model


class Guide_FM_Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = get_vector_field(args)

        self.basefm = load_base_vector_field(args)

        if self.args.use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema_rate)
            self.ema = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=avg_fn)
            for param in self.ema.parameters():
                param.requires_grad = False
        else:
            self.ema = None

        self.loss = nn.MSELoss(reduction='none')
        self.save_hyperparameters(logger=True)
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.debug = False

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if _has_basefm_segment(key):
                del state[key]
        metadata = getattr(state, "_metadata", None)
        if metadata is not None:
            for key in list(metadata.keys()):
                if _has_basefm_segment(key):
                    del metadata[key]
        return state

    def _add_basefm_to_state_dict(self, state_dict):
        if state_dict is None or any(key.startswith("basefm.") for key in state_dict):
            return state_dict
        state_dict = OrderedDict(state_dict)
        for key, value in self.basefm.state_dict().items():
            state_dict[f"basefm.{key}"] = value
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        state_dict = self._add_basefm_to_state_dict(state_dict) if strict else state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint):
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = self._add_basefm_to_state_dict(checkpoint["state_dict"])

    def on_save_checkpoint(self, checkpoint):
        if "state_dict" not in checkpoint:
            return
        for key in list(checkpoint["state_dict"].keys()):
            if _has_basefm_segment(key):
                del checkpoint["state_dict"][key]

    def forward(self, data, training=False):
        if not training and self.ema is not None:
            model = self.ema  # type: ignore
        else:
            model = self.model
        return model(data)

    def loss_fn(self, tr_pred, rot_pred, tor_pred, data):
        target_tr = getattr(data, 'guided_target_tr', data.u_tr).to(tr_pred.device).float()
        target_rot = getattr(data, 'guided_target_rot', data.u_rot).to(rot_pred.device).float()
        target_tor = getattr(data, 'guided_target_tor', data.u_tor)
        if target_tor is not None:
            target_tor = target_tor.to(tor_pred.device).float()

        w_tr = float(getattr(self.args, 'tr_weight', 1.0))
        w_rot = float(getattr(self.args, 'rot_weight', 1.0))
        w_tor = float(getattr(self.args, 'tor_weight', 1.0))
        loss_tr = self.loss(tr_pred, target_tr).sum(1).mean(0) * w_tr
        loss_rot = self.loss(rot_pred, target_rot).sum(1).mean(0) * w_rot
        if target_tor is None or tor_pred.numel() == 0:
            loss_tor = torch.zeros((), device=tr_pred.device)
        else:
            batch = torch.cat(
                [torch.full((int(n.item()),), i) for i, n in enumerate(data.num_torsions)]
            ).long().to(tor_pred.device)
            loss_tor = scatter(self.loss(tor_pred, target_tor), batch, dim=0, reduce='sum').mean(0) * w_tor
        loss = loss_tr + loss_rot + loss_tor
        return loss, loss_tr / max(w_tr, 1e-8), loss_rot / max(w_rot, 1e-8), loss_tor / max(w_tor, 1e-8)

    def _base_forward(self, data):
        with torch.no_grad():
            self.basefm.eval()
            return self.basefm(data)

    def training_step(self, data, batch_idx):
        batch_size = data.num_graphs
        tr_h, rot_h, tor_h = self(data, training=True)
        tr_base, rot_base, tor_base = self._base_forward(data)

        loss, tr_loss, rot_loss, tor_loss = self.loss_fn(
            tr_base + tr_h,
            rot_base + rot_h,
            tor_base + tor_h,
            data=data,
        )

        if not torch.isfinite(loss).all():
            self.debug = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            tr_loss = torch.nan_to_num(tr_loss, nan=0.0, posinf=0.0, neginf=0.0)
            rot_loss = torch.nan_to_num(rot_loss, nan=0.0, posinf=0.0, neginf=0.0)
            tor_loss = torch.nan_to_num(tor_loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.log('train/loss', loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/tr', tr_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/rot', rot_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/tor', tor_loss.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return {'loss': loss, 'tr_loss': tr_loss.detach(), 'rot_loss': rot_loss.detach(), 'tor_loss': tor_loss.detach()}

    def validation_step(self, data, batch_idx):
        batch_size = data.num_graphs
        tr_h, rot_h, tor_h = self(data, training=False)
        tr_base, rot_base, tor_base = self._base_forward(data)

        loss, tr_loss, rot_loss, tor_loss = self.loss_fn(
            tr_base + tr_h,
            rot_base + rot_h,
            tor_base + tor_h,
            data=data,
        )

        if not torch.isfinite(loss).all():
            self.debug = True
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            tr_loss = torch.nan_to_num(tr_loss, nan=0.0, posinf=0.0, neginf=0.0)
            rot_loss = torch.nan_to_num(rot_loss, nan=0.0, posinf=0.0, neginf=0.0)
            tor_loss = torch.nan_to_num(tor_loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/tr', tr_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/rot', rot_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val/tor', tor_loss.item(), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        return {'loss': loss, 'tr_loss': tr_loss.detach(), 'rot_loss': rot_loss.detach(), 'tor_loss': tor_loss.detach()}

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
                self.optimizer_zero_grad(epoch, batch_idx, optimizer)  # type: ignore
                break
        optimizer.step(closure=optimizer_closure)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update_parameters(self.model)  # type: ignore
