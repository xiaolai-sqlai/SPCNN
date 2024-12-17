_base_ = [
    '../_base_/models/mask-rcnn_spcnn_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='SPCNN',
        dims=[96,192,384,768],
        layers=[5,9,17,5],
        mlp_ratio=2.0,
        expand_ratio=1.0,
        drop_path_rate=0.40,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        init_cfg=dict(type='Pretrained', checkpoint="/data/laishenqi/TCSVT2024_SPCNN/checkpoint_base_8.5G.pth"),
        # init_cfg=dict(type='Pretrained', checkpoint="/data/laishenqi/TCSVT2024_SPCNN/checkpoint_base_256_11.1G.pth"),
    ),
    neck=dict(
        in_channels=[96,192,384,768],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
)

train_dataloader = dict(batch_size=4, num_workers=4)

# optimizer
optim_wrapper = dict(_delete_=True, type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.05), clip_grad=None)
