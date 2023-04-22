# Evaluator

  Evaluator for calculating the accuracy and resources consume. Accuracy
    calculation is optional

# TestFn

  A component of the evaluators, aimed to specify the accuracy evaluate step.

  Including:
  Acc TestFn : Normal accuracy TestFn,
  ZeroShot TestFn : Compute network zero-shot proxy score.
  Anticipate TestFn: Without searching/retraining stages, use acc_predictor(MLP/GP/RBF/CART) to predict a subnet's   performance, see more followed details


## Step0: Supernet pre-training on ImageNet

```bash
sh tools/slurm_train.sh $PARTION $JOB_NAME \
  configs/nas/bignas/bignas_mobilenetv3_large_supernet_32xb64.py \
  $WORK_DIR
```

## Step1: Predict the accuracy of a subnet sampled from supernet

  Define the type of the model in cfg:
```bash
  MLP(Multi Layer Perceptron):  predictor=dict(predictor_type='MLP', predictor_cfg=dict(n_layers=4), fig_cfg=dict(lr=1e-4))
  GP(Gaussian Process):   predictor=dict(predictor_type='GP')
  RBF(Radial Basis Function):  predictor=dict(predictor_type='RBF')
  CART(Classification and Regression Tree): predictor=dict(predictor_type='CART')
```
  You can spcify the optional parameters for predictor in predictor_cfg, like:
```bash
  predictor_cfg=dict(n_feature=78, n_layers=4, n_hidden=500, drop=0.1) for MLP
  predictor_cfg=dict(regr='constant', corr='cubic') for GP
  predictor_cfg=dict(kernel='cubic', tail='linear') for RBF
  predictor_cfg=dict(n_tree=500) for CART
```
  Specially for MLP, you can modify the training details with giving the fig_cfg, like:
```bash
  fig_cfg=dict(trn_split=0.7, lr=1e-4, epochs=1000)
```
  More details of these parameters is avaliable in gml/evaluators/predictor/


### With pretrained weights of accuracy predictor

 Add the pretrained weights' path in cfg, for example:
```bash
  MLP(Multi Layer Perceptron):  predictor=dict(predictor_type='MLP', pretrained = your_path_to_weights)
```
### Without pretrained weights

  Do nothing


### Specify your own training/evaluation samples
  You can retrain your subnets and use its final preformance to train the anticipate testfn, for example:
```bash
  MLP(Multi Layer Perceptron):  predictor=dict(predictor_type='MLP', pretrained = your_path_to_weights, train_samples = your_path_to_train_samples, evaluate_samples = your_path_to_evaluate_samples)
```
  Make sure giving a file with suffix '.yaml', in which contains key 'architectures' for a list of your subnets and key 'scores' for a list of each subnet's corresponding score.
  Check [bignas_500_samples.yaml](http://autolink.parrots.sensetime.com/pages/model/share/e5d49a82-3c62-4ecd-b93c-ffad41de45b3) for more information.

  If not given the pretrained path or training samples, you can modify this parameter to adjust total nums of samples in training stage
```bash
  predictor=dict(predictor_type='MLP', train_samples=50)
```

### Finally, start prediction:
```bash
sh tools/slurm_search.sh $PARTION $JOB_NAME \
  configs/nas/bignas/bignas_mobilenetv3_large_anticapate_testfn_8xb256.py
  $STEP0_CKPT --work-dir=$WORK_DIR
```

anticipate testfn will be automatically saved to your work-dir

|Search Config |  candidate_pool_size | Constraints(flops) | Best-Performance-Subnet |
|--------------- | --------------- |--------------- |--------------- |
|bignas_mobilenetv3_large_anticapate_testfn_8xb256.py  | 20   | 600M    | [subnet.yaml](http://autolink.parrots.sensetime.com/pages/model/share/bc42d46c-714a-4d84-92d7-46adb9193d36) / [search-log](http://autolink.parrots.sensetime.com/pages/model/share/a707e8fa-91fd-4fba-98ea-7a37d741592d)



## Step2: Evaluate the performance of the anticipate testfn

  Evaluate the anticipate testfn you got in STEP1

  To get the corralations between the prediction of candidates' performance and the their true performance,
  modify the cfg like this:

```bash
  MLP(Multi Layer Perceptron):  predictor=dict(predictor_type='MLP', pretrained = your_path_to_weights, evaluation = True)
```

  After this, you will get the Spearman's Rank-Order and the Kendall rank correlation coefficient

|Search Config |  candidate_pool_size | Constraints(flops) | Best-Performance-Subnet | Predictor's Performance |
|--------------- | --------------- |--------------- |--------------- | --------------- |
|bignas_mobilenetv3_large_anticapate_testfn_8xb256.py  | 20   | 600M    | [subnet.yaml](http://autolink.parrots.sensetime.com/pages/model/share/f43aa034-d621-4106-8fb7-4407756d9294) / [search-log](http://autolink.parrots.sensetime.com/pages/model/share/63af524f-fc9c-4a42-8399-e31616c59251) | Spearmans Rho (0.6333) Kendalls Tau (0.5000) |

  In case that you only want to get the subnet with the best performance , just modify the cfg like this:

```bash
  MLP(Multi Layer Perceptron):  predictor=dict(predictor_type='MLP', pretrained = your_path_to_weights, evaluation = False)
```

  After this, you will get the architecture of the best subnet and its performance


### Time Comparison

With train_samples = 20 for anticipate_testfn's training, candidate_pool_size = 20 for search step, search for 1 epoch.

|        Search  |  Predict w/o checkpoint | Predict w/ checkpoint |
|--------------- | --------------- |--------------- |
|  11 mins   |   11 mins     |        2 mins        |



### How to encoding the architecture:
  For a given architecture under a specific super net, we simply convert the choice of
each block into onehot vector(execpt for RBF, which we don't use onehot vector encoding) and concatenate all the vectors in the order of their block's name.


### Performance
Furthermore, we randomly sample 500 subnets and evaluate their preformance without retraining, we split 300 samples[(bignas_training_samples)](http://autolink.parrots.sensetime.com/pages/model/share/18c20b93-b6da-4d73-9be0-30f39145ae48) for anticipate_testfn's training and 200 samples[(bignas_evaluate_samples)](http://autolink.parrots.sensetime.com/pages/model/share/74b1124d-6f3a-49c7-a8bd-7390862284f4) for evaluation, the results of different anticipate_testfn's are as follows:

|     |  MLP | RBF | GP | CART
|--------------- | --------------- |--------------- |--------------- |--------------- |
|       RMSE      |   0.3602     |     0.3286       |     0.1874     |      0.4277         |
|  Spearmans Rho  |   0.8304     |      0.9179        |    0.9120    |      0.8813        |
|   Kendalls Tau  |   0.6410     |      0.7639      |    0.7625      |      0.7079        |
|   details  |   [evaluation-log](http://autolink.parrots.sensetime.com/pages/model/share/306e4ae9-8835-4ec4-b1e5-f08042c3ae28)    |     [evaluation-log](http://autolink.parrots.sensetime.com/pages/model/share/7512b04f-7c7a-46b0-a0ec-cb9286b24234)      |    [evaluation-log](http://autolink.parrots.sensetime.com/pages/model/share/5b20a470-42af-456b-ba5e-a0d2975698dd)      |      [evaluation-log](http://autolink.parrots.sensetime.com/pages/model/share/e7774b62-de82-405a-9236-4731f5e070b0)        |
