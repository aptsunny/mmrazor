# HPO guide to practice

## HPO 现有算法
HPO的算法主要体现在搜索算法上，包括`GridSearcher`, `BatchSearcher`, `RandomSearcher`，`SkoptSearcher`，`EvolutionSearcher`和`LamctsSearcher`。
### GridSearcher
#### 算法介绍
网格搜索优化算法，计算搜索空间笛卡尔积，无差别的选取配置用于训练。
#### 适用范围
支持所有搜索空间类型。


### BatchSearcher
#### 算法介绍
批优化算法，仅支持Nested类型，无优化计算，按序返回Nested中的每组配置。
#### 适用范围
仅支持Nested类型。

### RandomSearcher
#### 算法介绍
随机搜索优化算法，在搜索空间里随机搜索配置进行训练。
#### 适用范围
支持所有搜索空间类型。


### SkoptSearcher
#### 算法介绍
贝叶斯优化搜索算法，获取配置时，会根据已完成训练的历史结果进行优化，在搜索空间里寻找较优配置。
#### 适用范围
支持Int, Real, Categorical类型。

### EvolutionSearcher
#### 算法介绍
遗传变异优化算法。基于搜索空间随机生成种群，种群中包含population_size个初代个体。在每一代中随机选择两个父体进行交叉、变异产生新的个体。 调用get_config()获取个体时，优先选择初代个体，若无初代个体，会通过交叉变异产生新的下一代个体，所有个体都确保只生成一次，返回一次。

- 交叉：模拟染色体交叉操作，对两组父体配置进行部分段交换
- 变异：随机对配置进行改变（如增大学习率，减小Batch_size）。
#### 适用范围
支持所有搜索空间类型。

### LamctsSearcher
#### 算法介绍
LA-MCTS对搜索空间进行少量采样，训练Kmeans和SVM建树对空间进行分割，并使用基于UCB（upper confidence bound）的蒙特卡罗搜索，来平衡最优区间的开发和探索。
#### 适用范围
适合搜索空间连续，单任务耗时短的任务，如Mujoco强化学习任务。

## 算法所需资源（cpu资源）、精度
