from mmengine.config import read_base
with read_base():
    from .co_dino_5scale_r50_lsj_8xb2_1x_coco import *

param_scheduler.update([dict(milestones=[30])])
train_cfg.update(dict(max_epochs=36))
