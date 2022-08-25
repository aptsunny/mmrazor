_base_ = [
    'mmrazor::_base_/settings/imagenet_bs2048_autoslim_pil.py',
    'mmcls::_base_/default_runtime.py',
]

# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

supernet = dict(
    type='mmrazor.SearchableImageClassifier',
    data_preprocessor=_base_.preprocess_cfg,
    backbone=dict(type='mmrazor.AutoformerBackbone'),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmrazor.DynamicLinearClsHead',
        num_classes=1000,
        in_channels=640,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
)

num_samples = 2
model = dict(
    type='mmrazor.Autoformer',
    architecture=supernet,
    fix_subnet=None,
    mutators=dict(
        channel_mutator=dict(type='mmrazor.BigNASChannelMutator'),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')),
    data_preprocessor=data_preprocessor)

# model_wrapper_cfg = dict(
#     type='mmrazor.BigNASDDP',
#     broadcast_buffers=False,
#     find_unused_parameters=True)
# optim_wrapper = dict(accumulative_counts=num_samples + 2)

# learning policy
max_epochs = 100
param_scheduler = dict(end=max_epochs)

# train, val, test setting
train_cfg = dict(max_epochs=max_epochs)
# val_cfg = dict(type='mmrazor.AutoSlimValLoop')
