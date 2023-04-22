# 函数级调度 Functional Scheduling
函数级调度是pavi中的一个功能，用于并行调度执行函数，达到加速和利用闲时资源的目的。函数级调度的使用文档见[Functional Scheduling](http://al-sdkdoc.test.parrots.sensetime.com/func_scheduling.html)

## 函数级调度用于NAS model search stage
这里主要将model_searcher中的evaluate_subnet改为使用函数级调度执行。

### 配置环境、准备数据
同NAS demo中的[search-for-subnet-on-the-trained-supernet](https://gitlab.sz.sensetime.com/parrotsDL-sz/gml#step2-search-for-subnet-on-the-trained-supernet)

### 配置函数级调度环境
#### 配置pavi-queue
首先安装2.5.4以上版本的`pavi`
```bash
pip install --index-url https://pkg.sensetime.com/repository/pypi-proxy/simple/ --extra-index-url http://pavi.parrots.sensetime.com/pypi/simple/ --trusted-host pavi.parrots.sensetime.com -U --user pavi
```
以及1.0.4以上的[`pavi-queue`](http://al-sdkdoc.test.parrots.sensetime.com/queue.html)
```bash
pip install --index-url https://pkg.sensetime.com/repository/pypi-proxy/simple/ --extra-index-url http://pavi.parrots.sensetime.com/pypi/simple/ --trusted-host pavi.parrots.sensetime.com -U --user pavi-queue
```
#### 启动pavi-queue
启动
```bash
pavi queue.daemon -m --wxrobot
```
停止
```bash
pavi queue.stop
```
监控
```bash
pavi queue.hc
```
查看任务列表
```bash
pavi queue.list
```
其它指令可见[`pavi-queue`文档](http://al-sdkdoc.test.parrots.sensetime.com/queue.html)

#### 启动搜索任务
需要在搜索配置文件中，加入搜索参数和函数级调度资源参数`function_resource`
```python
searcher = dict(
    type='EvolutionModelSearcher',
    evaluator=dict(
        type='NaiveEvaluator', units='M', default_shape=(1, 3, 224, 224)),
    candidate_pool_size=4,
    candidate_top_k=1,
    constraints=dict(flops=600),
    metrics='accuracy',
    score_key='accuracy_top-1',
    max_epoch=20,
    num_mutation=2,
    num_crossover=2,
    mutate_prob=0.1,
    function_resource = dict(
        launcher='slurm',
        dist_params = dict(backend='nccl'),
        executor='cluster_sh1986-SH-IDC1-10-198-6-31',
        source='source /mnt/lustre/share/platform/env/pt1.7.0',
        partition='pat_dev',
        num_cpus=1,
        num_gpus=8,
        normal=True
    ))
```
其中各字段说明如下：
| 字段 | 说明 |
| ---- | ---- |
| `launcher` | 远程执行任务使用的launcher，一般为`slurm` |
| `dist_params` | 远程初始化环境所使用的参数 |
| `executor` | 远程执行任务所使用的集群，按照示例中的格式填写即可 |
| `source` | 在远程集群上所使用的环境，需要与主程序执行所依赖库的版本相同 |
| `partition` | 在远程集群上所使用的分区 |
| `num_cpus` | 函数每次执行所需要的cpu资源 |
| `num_gpus` | 函数每次执行所需要的gpu资源 |
| `normal` | 若为False，则为使用闲时任务调度，默认为`True` |

### 启动搜索任务
```bash
GPUS=1 GPUS_PER_NODE=1 tools/slurm_search.sh $PARTITION $JOB_NAME \
    ../bignas_mobilenetv3_large_supernet_32xb64/bignas_mobilenetv3\
    _large_supernet_32xb64.py $STEP1_CKPT --work-dir $WORK_DIR
```

## 函数级调度用于闲时任务占卡
这里，使用NAS find stage作为闲时任务占卡。首先，按照上面配置函数级调度环境和参数；然后将`tools/auto_schedule.py`中的cmd改为自己需要执行的find stage命令，最后执行
```bash
python auto_schedule.py
```
