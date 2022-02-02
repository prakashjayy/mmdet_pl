import pytorch_lightning as pl
from mmcv import Config
from mmdet.models import build_detector


class MMDET_TRAINER(pl.LightningModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        self.model.init_weights()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimzers(self):
        pass


if __name__ == "__main__":
    cfg = Config.fromfile("configs/faster_rcnn_r50_fpn.py")
    model = MMDET_TRAINER(cfg)

# DETECTORS.module_dict.keys() contains list of
# print(model)
