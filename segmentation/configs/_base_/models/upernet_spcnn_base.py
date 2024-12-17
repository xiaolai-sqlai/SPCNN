# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
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
    decode_head=dict(
        type='UPerHead',
        in_channels=[96,192,384,768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
