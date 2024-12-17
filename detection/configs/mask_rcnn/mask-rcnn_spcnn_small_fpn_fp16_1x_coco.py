_base_ = [
    '../_base_/models/mask-rcnn_spcnn_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        type='SPCNN',
        dims=[64,128,256,512],
        layers=[4,7,15,4],
        mlp_ratio=3.0,
        expand_ratio=1.0,
        drop_path_rate=0.30,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        init_cfg=dict(type='Pretrained', checkpoint="/data/laishenqi/TCSVT2024_SPCNN/checkpoint_small_4.4G.pth"),
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
)

train_dataloader = dict(batch_size=4, num_workers=4)

# optimizer
optim_wrapper = dict(_delete_=True, type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.05), clip_grad=None)
