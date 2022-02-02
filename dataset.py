from dataclasses import dataclass
from functools import partial

import pytorch_lightning as pl
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmdet.datasets import build_dataset
from mmdet.datasets.dataset_wrappers import ConcatDataset


@dataclass
class MMDET_LOADER(pl.LightningDataModule):
    cfg: Config
    stage: int = None

    def __post_init__(self):
        super().__init__()
        self.setup(self.stage)
        if self.cfg.gpus is not None:
            count = (
                self.cfg.gpus
                if isinstance(self.cfg.gpus, int)
                else len(self.cfg.gpus)
            )  # mostly list of gpus
        else:
            # No GPUs are being used.
            count = 1
        self.batch_size = self.cfg.data.samples_per_gpu * count
        self.num_workers = self.cfg.data.workers_per_gpu * count

    def setup(self, stage=None):
        # 'img_metas', 'img', 'gt_bboxes', 'gt_labels'
        if isinstance(self.cfg.data.train, (list, tuple)):
            self.train_ds = ConcatDataset([build_dataset(c) for c in cfg])
        else:
            self.train_ds = build_dataset(self.cfg.data.train)
        self.val_ds = build_dataset(self.cfg.data.val, dict(test_mode=True))
        # TODO: Add Test dataset too

    def train_dataloader(self):
        # depending on EpochBasedRunner or IterBasedRunner: shuffle or not, we have to select a batch_sampler.
        # For now we will use EpochBasedRunner. #TODO IterBasedRunner: check build_data_loader in mmdetection
        # we have also ignored the runner key from config file. runner = dict(type='EpochBasedRunner', max_epochs=24)

        # For reasons mentioned here,
        # https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565
        # we will be using distributed data parallel always for training both for
        # multi-node (multiple machines with every machine having multiple GPUs)
        # and single node (one machine with multiple GPUs)

        # which sampler to use DistributedGroupSampler, DistributedSampler or GroupSampler?
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=partial(collate, samples_per_gpu=self.batch_size),
            pin_memory=False,
        )

    def val_dataloader(self):
        pass


if __name__ == "__main__":
    cfg = Config.fromfile("configs/faster_rcnn_r50_fpn.py")
    ds = MMDET_LOADER(cfg)
    for block in ds.train_dataloader():
        # dict_keys(['img_metas', 'img', 'gt_bboxes', 'gt_labels'])
        img_metas = block["img_metas"].data[0]
        imgs = block["img"].data[0]
        gt_bboxes = block["gt_bboxes"].data[0]
        print(
            f"img_metas: {len(img_metas)} images: {imgs.shape}, gt_bboxes: {[i.shape for i in gt_bboxes]}"
        )
