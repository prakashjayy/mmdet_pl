import pytorch_lightning as pl
import torch
from mmcv import Config
from mmdet.models import build_detector
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


def build_optimizer(cfg, model):
    type = cfg.pop("type")
    optim = getattr(torch.optim, type)
    return optim(**cfg, params=model.parameters())


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db
        and not any(x in k for x in exclude)
        and v.shape == db[k].shape
    }


class MMDET_TRAINER(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        self.model.init_weights()

    def forward(self, x):
        result = self.model(return_loss=False, rescale=True, **x)
        return result

    def training_step(self, batch, batch_idx):
        batch = {k: v.data[0] for k, v in batch.items()}
        losses = self.model.train_step(batch, None)
        for loss_type, loss_value in losses["log_vars"].items():
            if "loss" in loss_type:
                self.log(f"train_{loss_type}", loss_value, prog_bar=True)
        self.log("train_acc", losses["log_vars"]["acc"], prog_bar=True)
        return {"loss": losses["loss"]}

    def validation_step(self, batch, batch_idx):
        batch = {k: v.data[0] for k, v in batch.items()}
        losses = self.model.val_step(batch, None)
        self.log("val_loss", losses["loss"], prog_bar=True)
        for loss_type, loss_value in losses["log_vars"].items():
            if "loss" in loss_type:
                self.log(f"val_{loss_type}", loss_value, prog_bar=True)
        self.log("val_acc", losses["log_vars"]["acc"], prog_bar=True)
        return {"loss": losses["loss"]}

    def configure_optimizers(self):
        # optimizer
        optimizer = build_optimizer(self.cfg.optimizer, self.model)

        # scheduler
        lr_config = self.cfg.lr_config
        if lr_config is not None:
            # TODO call multisteplr based on config file.
            lr_hook = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=lr_config.step
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_hook,
                    "interval": lr_config.policy,
                },
            }
        return optimizer


def main(cfg, model, dl):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        filename="{epoch}-{step}-{val_loss:.3f}",
        save_last=True,
    )

    if hasattr(cfg, "load_from") & (cfg.load_from is not None):
        print("[Loading model from %s]" % cfg.load_from)
        weights = torch.load(cfg.load_from)
        csd = intersect_dicts(weights["state_dict"], model.state_dict())
        model.load_state_dict(csd, strict=False)

    trainer = pl.Trainer(
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=2,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=2)],
        logger=pl.loggers.TensorBoardLogger(
            f"lightning_logs/{cfg.exp_name}", name="integrated"
        ),
    )

    trainer.fit(model, dl)


if __name__ == "__main__":
    from dataset import MMDET_LOADER

    cfg = Config.fromfile("configs/faster_rcnn_r50_fpn.py")

    # Load model
    model = MMDET_TRAINER(cfg)
    dl = MMDET_LOADER(cfg)
    main(cfg, model, dl)
