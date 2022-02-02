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