# mmdet_pl
using pytorch lightning for training object detection models in mmdet. 

## Usage
- clone the repo
- do `poetry install` 
- To contribute, also do `pre-commit install` 

## Functions 
- Loading a mmcv config file.
```python
from mmcv import Config 
cfg = Config.fromfile("configs/faster_rcnn_r50_fpn.py")
```

## Implementation
= [x] dataset and dataloader
= [x] define model
- [ ] optimizers setup 
- [ ] training_step and logger
- [ ] validation_step and logger
- [ ] validation_epoch_end (coco utils checks)
- [ ] train the model for few epochs
- [ ] write inference pipeline
- [ ] write test pipeline to calculate the stats. 