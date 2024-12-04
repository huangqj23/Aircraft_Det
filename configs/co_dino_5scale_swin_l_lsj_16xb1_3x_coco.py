from mmengine.config import readbase
with read_base():
    from .co_dino_5scale_swin_l_lsj_16xb1_1x_coco import *

model.update(
    dict(backbone=dict(drop_path_rate=0.5))
)

param_scheduler.update(
    [dict(type='MultiStepLR', milestones=[30])]
)

train_cfg.update(
    dict(max_epochs=36)
)
