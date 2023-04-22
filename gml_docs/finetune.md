## Use GML Searchable Backbone

我们以一个检测任务为例，我们将会选用FasterRCNN作为我们的基础检测模型。

回顾MMDetection 中的Faster RCNN检测效果:
|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  caffe  |   1x    | -        | -              | 37.2   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909.log.json) |
|    R-50-FPN     |  caffe  |   1x    | 3.8      |                | 37.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_20200504_180032.log.json) |
|    R-50-FPN     | pytorch |   1x    | 4.0      | 21.4           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
| R-50-FPN (FP16)     | pytorch | 1x      | 3.4      | 28.8           | 37.5   |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/faster_rcnn_r50_fpn_fp16_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204_143530.log.json) |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 38.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_20200504_210434.log.json) |
|    R-101-FPN    |  caffe  |   1x    | 5.7      |                | 39.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_20200504_180057.log.json) |
|    R-101-FPN    | pytorch |   1x    | 6.0      | 15.6           | 39.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130_204655.log.json) |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 39.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_20200504_210455.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    | 7.2      | 13.8           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203_000520.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_20200506_041400.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    | 10.3     | 9.4            | 42.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204_134340.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    | -        | -              | 41.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033.log.json)  |

这些以ResNet为backbone的模型能够提供极高的检测精度，但是他们的计算速度太慢了；对于一些落地场景，模型推理时间也是一个极为重要的选型指标，通过对限制条件进行规约进行进行多目标优化，我们可以使用GML工具箱来得到一个轻量、快速、高精度的FasterRCNN模型。在GML里使用mmdetection来实现一个FasterRCNN模型，我们首先需要安装mmdetection，同时由于mmdetection中涉及到cuda编程，因此我们需要安装mmcv-full版本，mmdtection和mmcv版本兼容性可以在[这里](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/get_started.md)找到。注意，如果已经安装了 mmcv，首先需要使用 pip uninstall mmcv 卸载已安装的 mmcv，如果同时安装了 mmcv 和 mmcv-full，将会报 ModuleNotFoundError 错误。假设当前已经成功安装 CUDA 10.1，这里提供了一个基于 conda 安装 MMDetection 的脚本。
```bash
pip install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

下面我们以bignas算法搜索出来的模型作为backbone，在GML里面实现一个轻量、快速、高精度的FasterRCNN模型。

- 我们首先到GML仓库下configs/nas/bignas/README位置下载bignas提供的在ImageNet数据集上做分类的模型以及对应的子网的yaml文件，我们以bignas提供的Flops250M为例，下载对应权重及yaml文件。
- 在configs/bignas/文件夹下新建bignas_faster_rcnn_fpn_coco_detection.py配置文件，然后我们将mmdetection/config/_base_/models/faster_rccn_r50_fpn.py文件内容拷贝到新建的配置文件里，然后对model配置做如下修改：

### step1: 新增如下字段
```python
'''
coco_detection.py: coco数据集的配置，包含训练流程，测试流程
schedule_1x.py：训练时使用的优化器、学习率、以及学习率衰减策略
mmdet_runtime.py：训练时checkpoint hook以及logger hook等
'''
_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_1x.py',
    '../../_base_/mmdet_runtime.py'
]
```
### step2: 针对model做如下修改：
model的type属性从FasterRCNN改为'mmdet.FasterRCNN'，因为跨仓库调用时需要加上mmdet.
替换backbone,即复制同文件夹下bignas_mobilenetv3_large_supernet_32xb64.py中的model里的backbone的配置，并修改以下内容：新增with_last_layer=False, 因为做检测我们不需要最后一层卷积，只需要用backbone提取多级特征即可；新增`out_indices=[1,2,4,6]`，即提取MobileNetV3Slice中的第二、三、五、六层特征作为FPN的输入；将norm_cfg改为`norm_cfg = dict(type='SyncBatchNormSlice', requires_grad=True)`，这是因为在微调过程中，BN的参数被固定为训练前的批统计量。但在Bignas中，由于归一化的特征在不同路径上是不相等的，因此冻结BN是不可行的。另一方面，与图像分类不同，目标检测器是用高分辨率图像训练的。由于内存的限制，这导致了小批量的结果，并严重降低了BN的准确性。因此在训练过程中，我们用同步批处理归一化(SyncBN)代替传统的BN。它可以跨多个gpu计算批统计数据，并增加有效的批大小。
```python
norm_cfg = dict(type='SyncBatchNormSlice', requires_grad=True)
model = dict(
    type='mmdet.FasterRCNN',
    backbone=dict(
        type='gml.MobileNetV3Slice',
        conv_cfg=dict(type='Conv2dSlice'),
        norm_cfg=norm_cfg,
        with_last_layer=False,
        out_indices=[1,2,4,6],
        search_space=dict(
            MB_stage_blocks=[
                dict(type='Categorical', data=[1],
                     default=1),  # first conv, place_holder
                dict(
                    type='Categorical',
                    data=[1, 2],
                    default=1,
                    key='MB_stage_blocks.1'),
                dict(
                    type='Categorical',
                    data=[3, 4, 5],
                    default=3,
                    key='MB_stage_blocks.2'),
                dict(
                    type='Categorical',
                    data=[3, 4, 5, 6],
                    default=3,
                    key='MB_stage_blocks.3'),
                dict(
                    type='Categorical',
                    data=[3, 4, 5, 6],
                    default=3,
                    key='MB_stage_blocks.4'),
                dict(
                    type='Categorical',
                    data=[3, 4, 5, 6, 7, 8],
                    default=3,
                    key='MB_stage_blocks.5'),
                dict(
                    type='Categorical',
                    data=[3, 4, 5, 6, 7, 8],
                    default=3,
                    key='MB_stage_blocks.6'),
                dict(
                    type='Categorical',
                    data=[1, 2],
                    default=1,
                    key='MB_stage_blocks.7'),
                dict(
                    type='Categorical',
                    data=[1],
                    default=1,
                    key='MB_stage_blocks.8'),  # last conv
            ],
            MB_kernel_size=[
                dict(
                    type='Categorical',
                    data=[3],
                    default=3,
                    key='MB_kernel_size.0'),  # first conv
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.1'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.2'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.3'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.4'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.5'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.6'),
                dict(
                    type='Categorical',
                    data=[3, 5],
                    default=3,
                    key='MB_kernel_size.7'),
                dict(
                    type='Categorical',
                    data=[1],
                    default=1,
                    key='MB_kernel_size.8'),  # last conv
            ],
            MB_expand_ratio=[
                dict(type='Categorical', data=[1],
                     default=1),  # first conv, place_holder
                dict(
                    type='Categorical',
                    data=[1],
                    default=1,
                    key='MB_expand_ratio.1'),
                dict(
                    type='Categorical',
                    data=[4, 5, 6],
                    default=4,
                    key='MB_expand_ratio.2'),
                dict(
                    type='Categorical',
                    data=[4, 5, 6],
                    default=4,
                    key='MB_expand_ratio.3'),
                dict(
                    type='Categorical',
                    data=[4, 5, 6],
                    default=4,
                    key='MB_expand_ratio.4'),
                dict(
                    type='Categorical',
                    data=[4, 5, 6],
                    default=4,
                    key='MB_expand_ratio.5'),
                dict(
                    type='Categorical',
                    data=[6],
                    default=6,
                    key='MB_expand_ratio.6'),
                dict(
                    type='Categorical',
                    data=[6],
                    default=6,
                    key='MB_expand_ratio.7'),
                dict(
                    type='Categorical',
                    data=[6],
                    default=6,
                    key='MB_expand_ratio.8'),  # last conv
            ],
            MB_out_channels=[
                dict(
                    type='Categorical',
                    data=[16, 24],
                    default=16,
                    key='MB_out_channels.0'),  # noqa: E501 first conv
                dict(
                    type='Categorical',
                    data=[16, 24],
                    default=16,
                    key='MB_out_channels.1'),
                dict(
                    type='Categorical',
                    data=[24, 32],
                    default=24,
                    key='MB_out_channels.2'),
                dict(
                    type='Categorical',
                    data=[32, 40],
                    default=32,
                    key='MB_out_channels.3'),
                dict(
                    type='Categorical',
                    data=[64, 72],
                    default=64,
                    key='MB_out_channels.4'),
                dict(
                    type='Categorical',
                    data=[112, 120, 128],
                    default=112,
                    key='MB_out_channels.5'),
                dict(
                    type='Categorical',
                    data=[192, 200, 208, 216],
                    default=208,
                    key='MB_out_channels.6'),  # noqa: E501
                dict(
                    type='Categorical',
                    data=[216, 224],
                    default=224,
                    key='MB_out_channels.7'),
                dict(
                    type='Categorical',
                    data=[1792, 1984],
                    default=1984,
                    key='MB_out_channels.8'),  # noqa: E501 last conv
            ])),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 112, 224],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
```
### step3: 新增algorithm字段，原来的build_model变为build_algorithm，同时原来的model成为algorithm.model

```python
# 新增algorithm字段，原来的build_model变为build_algorithm
algorithm = dict(
    type='gml.BigNAS',
    model=model,
    retraining=True,
    strategy='sandwish4',
    drop_ratio=0.2,
    drop_path_ratio=0.2,
    grad_clip=dict(clip_type='by_value', clip_value=1.0),
    mutator=dict(type='gml.StateslessMutator', fixed_mutable_cfg=None),
    distiller=None)
find_unused_parameters=True
```

### step4: 启动任务
```bash
sh tools/slurm_train.sh $PARTION $JOB_NAME \ configs/nas/bignas/bignas_faster_rcnn_fpn_coco_detection.py $WORK_DIR \--cfg-options algorithm.mutable_cfg=$DOWNLOADED_SUBNET_YAML load_from=$DOWNLOADED_CKPT
```

由此我们完成了基于FasterRCNN算法的backbone的替换，实验结果请参考此[文档](../projects/downstream_tasks/bignas_object_detection/README.md)
