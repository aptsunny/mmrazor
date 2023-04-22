# Transfer learning supernet to other tasks.

## 利用超网在下游任务进行搜索
我们利用在imagenet上训练的超网为例在下游任务进行搜索，从而找到最适合下游任务的框架。这种方法一般包括三个步骤:ImageNet上的超网预训练，下游任务数据集上的超网微调，以及使用进化算法对训练好的超网进行搜索。具体的训练步骤可以参考detnas的[流程](../configs/nas/detnas/README.md)。一般来说，在下游任务上直接搜索的网络比在ImageNet分类上搜索的网络具有一致的优势，但是这些步骤是耗时的，我们以DetNAS为例展示该方法三个步骤的消耗时间：

|Supernet pre-training|Supernet fine-tuning|Search on the supernet|
|:---------------:|:---------------:|:---------------:|
|3 x $10^5$ iterations|9 x $10^4$ iterations|20 x 50 models|
|8 GPUS on 1.5days| 8 GPUs on 1 days|8 GPUs on 1.5 day|

## 利用在ImageNet上搜索的网络结构直接作为backbone应用到下游任务
我们利用在ImageNet上搜索的网络结构直接作为backbone应用到下游任务，这种方法可以节省大量的时间，但是用于下游任务的网络与用于分类的网络在视觉上是不同的，因此直接将ImageNet上训练得到的backbone应用到下游任务可能是次优的。具体的流程可以参考[此文](./finetune.md)。同时我们以bignas为例给出了bignas在imagenet上训练得到的backbone应用到下游实验任务取得的精度以及消耗的时间。

### Bignas on object detection
| Dataset | Model | config |Lr schd| trained model | log | pavi |Flops(M)|mAP|Consuming time|
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|COCO|bignas_faster_rcnn_250M|[config](./bignas_faster_rcnn_fpn_coco_detection_250M.py)|1x|[model](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[log](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[pavi](http://autolink.parrots.sensetime.com/pages/training/share/c70596f6-c383-463f-8587-e20edc6afc02)|127937|36.9|8 GPUs on 1 days
|COCO|bignas_faster_rcnn_350M|[config](./bignas_faster_rcnn_fpn_coco_detection_350M.py)|1x|[model](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[log](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[pavi](http://autolink.parrots.sensetime.com/pages/training/share/b00f5a91-9219-49c8-a230-0f82bd147d5b)|128063|37.0|8 GPUs on 1 days
|COCO|bignas_faster_rcnn_600M|[config](./bignas_faster_rcnn_fpn_coco_detection_600M.py)|1x|[model](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[log](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[pavi](http://autolink.parrots.sensetime.com/pages/training/share/9d22b842-2778-4852-a5cb-a36825b11182)|130328|38.2|8 GPUs on 1 days
|COCO|bignas_faster_rcnn_1000M|[config](./bignas_faster_rcnn_fpn_coco_detection_1000M.py)|1x|[model](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[log](http://autolink.parrots.sensetime.com/pages/model/share/691784ab-e2c1-4596-a231-201e772ba0b5)|[pavi](http://autolink.parrots.sensetime.com/pages/training/share/bbceebc9-696c-47dc-93e0-61ea6bc80518)|133376|39.1|8 GPUs on 1 days

### Bignas on pose estimation
|Dataset|Model|config|Input Size|trained model|log|pavi|Flops(M)|AP|Consuming time|
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|COCO|Topdown_bignas_250M| [config](./bignas_topdown_coco_256x192_250M.py) |256x192| [model](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [log](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [pavi](http://autolink.parrots.sensetime.com/pages/compare/share/f32ac19f-a8b8-4fad-935d-88c33a226a47)|339|66.08|8 GPUs on 1.5 days
|COCO|Topdown_bignas_350M| [config](./bignas_topdown_coco_256x192_350M.py) |256x192| [model](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [log](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [pavi](http://autolink.parrots.sensetime.com/pages/compare/share/f32ac19f-a8b8-4fad-935d-88c33a226a47)|345|66.23|8 GPUs on 1.5 days
|COCO|Topdown_bignas_600M| [config](./bignas_topdown_coco_256x192_600M.py) |256x192| [model](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [log](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [pavi](http://autolink.parrots.sensetime.com/pages/compare/share/f32ac19f-a8b8-4fad-935d-88c33a226a47)|449|67.38|8 GPUs on 1.5 days
|COCO|Topdown_bignas_1000M| [config](./bignas_topdown_coco_256x192_1000M.py) |256x192| [model](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [log](http://autolink.parrots.sensetime.com/pages/model/share/fdd0c9dc-f00f-40d4-8632-ed05ed9f2c94) | [pavi](http://autolink.parrots.sensetime.com/pages/compare/share/f32ac19f-a8b8-4fad-935d-88c33a226a47)|591|68.42|8 GPUs on 1.5 days

### Bignas on semantic segmentation
|Dataset|Model|config|Crop Size|Lr schd|trained model|log|pavi|Flops(M)|mIoU|Consuming time
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|Cityscapes|DeepLabV3plus_bignas_250M| [config](./bignas_deeplabv3plus_512x1024_40k_cityscapes_250M.py) |512x1024|40000| [model](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [log](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [pavi](http://autolink.parrots.sensetime.com/pages/training/share/986ff765-6847-4c41-adc2-93fc1fa4c3b4)|28373|69.45|8 GPUs on 1 days
|Cityscapes|DeepLabV3plus_bignas_350M| [config](./bignas_deeplabv3plus_512x1024_40k_cityscapes_350M.py) |512x1024|40000| [model](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [log](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [pavi](http://autolink.parrots.sensetime.com/pages/training/share/8c8175bf-df21-462f-bfe0-a6f40e99fad8)|28427|68.54|8 GPUs on 1 days
|Cityscapes|DeepLabV3plus_bignas_600M| [config](./bignas_deeplabv3plus_512x1024_40k_cityscapes_6000M.py) |512x1024|40000| [model](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [log](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [pavi](http://autolink.parrots.sensetime.com/pages/training/share/64ac6b9e-8c71-455b-bcfe-2de542a3598b)|29527|71.45|8 GPUs on 1 days
|Cityscapes|DeepLabV3plus_bignas_1000M| [config](./bignas_deeplabv3plus_512x1024_40k_cityscapes_1000M.py) |512x1024|40000| [model](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [log](http://autolink.parrots.sensetime.com/pages/model/share/ebea47de-a54f-4b0c-b2ef-af7418076eeb) | [pavi](http://autolink.parrots.sensetime.com/pages/training/share/97c204eb-b00b-4555-867d-56f762f61921)|31015|73.2|8 GPUs on 1 days

## 多任务代理搜索
多任务代理搜索与进化搜索算法的区别是:首先，多任务代理搜索直接继承了在ImagneNet上搜索的权值，从而可以跳过花费大量时间的超网预训练步骤。其次，该方法将对候选数据进行抽样，并在给定的新数据集上进行训练，从而使anticipate_testfn得到更精确的分数，从而引导多任务代理搜索算法找到更理想的候选数据。最后，该方法使用多目标算法，以平衡指定的score key(accuracy-top-1)和第二个对象(flops, latency…)来搜索最佳候选，这与每次搜索步骤总是选择最准确的候选不同。总之，多任务代理搜索可以处理任何新数据集中的搜索步骤，并将返回最满意的候选者，即score key和第二个对象之间的最优解。如果你想在下游任务中使用多任务代理搜索(如对象检测、实例分割)，只需将搜索cfg中的score_key更改为指定的任务score_key和trade_off中的max_score_key即可。
