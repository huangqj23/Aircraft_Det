from mmengine.config import read_base
from mmdet.models.backbones import SwinTransformer

with read_base():
    from .co_dino_5scale_r50_lsj_8xb2_1x_aircraft import *

pretrained = '/data1/huangqj/Aircraft_Det/pretrained/swin_large_patch4_window12_384_22k.pth'  # noqa
    
# model settings
model.update(
    dict(
    data_preprocessor=dict(batch_augments=batch_augments),
    backbone=dict(
        type=SwinTransformer,
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5, # 0.3 -> 0.5
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type=PretrainedInit, checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(transformer=dict(encoder=dict(with_cp=6)))))

param_scheduler=[dict(type='MultiStepLR', milestones=[30])]


train_cfg.update(
    dict(max_epochs=36)
)