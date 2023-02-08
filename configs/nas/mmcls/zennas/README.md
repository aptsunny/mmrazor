# Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition


## Abstract

Zen-NAS is a lightning fast, training-free Neural Architecture Searching (NAS) algorithm for automatically designing deep neural networks with high prediction accuracy and high inference speed on GPU and mobile device.

Using 1 GPU searching for 12 hours, ZenNAS is able to design networks of ImageNet top-1 accuracy comparable to EfficientNet-B5 (\~83.6%) while inference speed 4.9x times faster on V100, 10x times faster on NVIDIA T4, 1.6x times faster on Google Pixel2.

![Inference Speed](/docs/imgs/model_zoo/zennas/ZenNet_speed.png)

## Apply on shufflenet search space

### Step1: Searching with Zero
```bash
sh tools/slurm_search.sh $PARTION $JOB_NAME \
configs/nas/zennas/mobilenetv3_zenscore_evolution_search_8xb256.py $STEP1_CKPT --work-dir $WORK_DIR
```

### Step2: Subnet test on ImageNet

```bash
sh tools/slurm_test.sh $PARTION $JOB_NAME \
configs/nas/zennas/bignas_mobilenetv3_large_subnet_8xb128_calibBN.py $STEP2_CKPT --work-dir $WORK_DIR --eval accuracy --cfg-options algorithm.mutable_cfg=$STEP2_YAML
```

## Guide to Practice
1. 候选网络结构通过`gml.GeNet`的backbone通过参数加载实现;
2. XX 即为对应latency.
3. 根据配置文件`mobilenetv3_zenscore_evolution_search_8xb256.py`测试在bignas的搜索空间下，zen-score指标如下:

| search space | flops(M) | Capacity(M) |  zen-score  | acc-top1 |
| --------   | -----:  |-----:  | -----:  |  -----:  |
| mobilenetV3_600M  |   Flops:585.138 |  Capacity:18.626 |  118.75383758544922 | 78.26600646972656 |

## Citation

```latex
@inproceedings{ming_zennas_iccv2021,
  author    = {Ming Lin and Pichao Wang and Zhenhong Sun and Hesen Chen and Xiuyu Sun and Qi Qian and Hao Li and Rong Jin},
  title     = {Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition},
  booktitle = {2021 IEEE/CVF International Conference on Computer Vision, {ICCV} 2021},
  year      = {2021},
}
```

```python
_base_ = ['./zennas_plainnet_supernet_8xb128_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.ZeroShotLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=480000,
    population_size=512,
    num_classes=1000,
    no_create=False,
    # estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='Zen')
    # estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='xd')
    # estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='GradNorm')
    # estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='NASWOT') # 对模型中间的RELU
    estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='TE-NAS')
)
```