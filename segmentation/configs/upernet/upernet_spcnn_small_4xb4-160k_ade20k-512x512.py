_base_ = [
    '../_base_/models/upernet_spcnn_small.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

optim_wrapper = dict(_delete_=True, type='AmpOptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005), clip_grad=None)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=160000, eta_min=0.0, by_epoch=False,)
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))
