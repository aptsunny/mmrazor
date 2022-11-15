default_scope = 'mmcls'

# !dataset config
# ==========================================================================
use_mc = True
if use_mc:
    file_client_args = dict(
        backend='memcached',
        server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
        client_cfg='/mnt/lustre/share/memcached_client/client.conf')
else:
    file_client_args = dict(backend='disk')

dataset_type = 'ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=preprocess_cfg['mean'],
    std=preprocess_cfg['std'],
    # convert image from BGR to RGB
    bgr_to_rgb=preprocess_cfg['to_rgb'],
)

bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # TODO(shiguang): align with gml.AutoAugmentV2
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),  # , backend='pillow'),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
# ==========================================================================
# !schedule config
# ==========================================================================
# ! grad clip by value is not supported!
num_samples = 2
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00001),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.), # 
    accumulative_counts=num_samples + 2) # num_samples + 2

# learning policy
max_epochs = 360
warmup_epochs = 5
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - warmup_epochs,
        eta_min=0,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1) # -> build_train_loop
val_cfg = dict(type='mmrazor.AutoSlimValLoop', calibrated_sample_nums=4096)
test_cfg = dict(type='mmrazor.AutoSlimTestLoop', calibrated_sample_nums=4096)

# auto_scale_lr = dict(base_batch_size=2048)