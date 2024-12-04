from mmengine.config import read_base

with read_base():
    from .co_dino_5scale_swin_l_16xb1_1x_coco import *

# model settings
model.update(dict(backbone=dict(drop_path_rate=0.6)))

param_scheduler.update([dict(milestones=[30])])
train_cfg.update(dict(max_epochs=36))
