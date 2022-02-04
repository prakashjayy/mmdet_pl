# mmdet_pl
using pytorch lightning for training object detection models in mmdet. 

- mmdet implementation of datasets and network is cool but going through their training/inference code is a mess (atleast for me). so this repo tries to decorate code using pytorch lightning. 
- This is still experimental and contains lot of bugs. For instance. mmdet uses an iterbased or epochbased learner and they have optimizer and schedular defined accordingly. I have made some changes here as I didn't find a way to embed an iter+epoch (steplr(epoch) + warmup(iter)) based schedular in pytorch lightning yet. 

## Usage
- clone the repo
- do `poetry install` 
- To contribute, also do `pre-commit install` 


## Implementation
- [x] dataset and dataloader
- [x] define model
- [x] optimizers setup 
- [x] training_step and logger
- [x] validation_step and logger
- [ ] validation_epoch_end (coco utils checks)
- [x] train the model for few epochs
- [ ] write inference pipeline
- [ ] write test pipeline to calculate the stats. 