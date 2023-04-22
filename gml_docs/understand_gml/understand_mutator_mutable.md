# Mutable
Mutables是GML算法中一个基础的可搜索层，在GML的剪枝、NAS、量化算法中，我们往往需要在训练的过程中动态的调整某些层。⽐如NAS⾥需要搜索某些层的类型(MixedOp)、kernel_size（SliceOp）；剪枝算法⾥需要搜索通道数（SliceOp）；量化算法⾥需要搜索Bits数（SliceOp）。
Mutables仅包含必要操作的权重，比如候选操作本身的权重，至于其他如何调整Mutables的属性如调整架构的状态和权重都放在Mutator中。Mutables有一个key属性，它标记可变的标识，用户可以使用这个标识在网络的各个地方进行共享决策，在Mutator的实现中，Mutator应该使用这个key来区分不同的变量。共享相同key的变量应该彼此“相似”。目前key的默认作用域是全局的。默认情况下，键使用从1开始的全局计数器来产生唯一的id。

在我们的设计中， Mutable包含以下两类：
- MixedOp: 由多个候选op组成，候选op为nn.Module,在网络的每一次forward过程中它会从多个候选op中选择一个然后将其应用到输入上并返回结果。MixedOp有如下属性/方法:

1. `__getitem__()`方法，调用该方法后能通过索引得到MixedOp中某个候选op；
2. `__setitem__()`方法，调用该方法后能对MixedOp中候选op进行修改；
3. `__delitem__()`方法，调用该方法后能够删除掉MixedOp中某个候选op；
4. `__len__()`方法，调用该方法后能返回MixedOp中候选op的个数；
5. `__iter__()`方法，调用该方法后使MixedOp成为可迭代对象；
6. `set_mutator()`方法，为当前Mutable设置Mutator，从而可以进行网络forward过程；
7. `export()`方法，调用该方法后能使得可变op组合变为固定的op，这常在搜索阶段和最终子网重训阶段使用；

**MixedOp使用的一个例子**

```python
import random
from collections import OrderedDict

import torch
import torch.nn as nn

from gml.models import MixedOp

class DummyMutator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward_mixed_op(self, mutable, *args, **kwargs):
        idx = random.randint(0, len(mutable) - 1)
        out = mutable[idx](*args, **kwargs)
        return out, None

op_candidates = OrderedDict([('conv3x3', nn.Conv2d(16, 128, 3)),
                             ('conv5x5', nn.Conv2d(16, 128, 5)),
                             ('conv7x7', nn.Conv2d(16, 128, 7))])
layer = MixedOp(op_candidates, key='layer_search')
mutator = DummyMutator()
layer.set_mutator(mutator)
x = torch.randn(2, 16, 16, 16)
output = layer(x)
export_output = export_layer(x)
assert torch.equal(layer[2](x), export_output)
```
- SliceOP:SliceOp是一个可切片的Op，包含一个可以切片权重的超级内核。与MixedOp不同的是，候选对象必须具有相同的类型，并与本地权重共享。在每个forward中，它选择一个候选对象，切片相应的权重，应用于它们的公共函数(例如:torch.nn.function.conv2d)。注意:本地权重是在构建阶段使用search_space中的“max”值分配的。因此，只能缩小search_space。SliceOP有如下属性/方法:

1. `get_value()`方法，根据值的类型和种类得到相应值
2. `forward_inner`方法，每个slice op应该实现这个函数使得Mutator内部的forward_slice_op可以调用这个函数.

**SliceOP使用的一个例子**

```python
from gml.models import Conv2dSlice
# Please refer to understand_space to learn about search space!
from gml.space import Categorical, Int, choice_to_space

class DummyMutator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward_mixed_op(self, mutable, *args, **kwargs):
        idx = random.randint(0, len(mutable) - 1)
        out = mutable[idx](*args, **kwargs)
        return out, None

args = dict(in_channels=6,
            out_channels=Int(9, 28, step=18),
            kernel_size=Categorical(data=[3, 5, 7]),
            groups=3,
            key='conv_search')
cfg = dict(type='Conv2dSlice', padding=None)
layer = build_conv_layer(cfg, **args)

mutator = DummyMutator()
layer.set_mutator(mutator)
x = torch.rand(2, 6, 128, 128)
och = layer.out_channels(kind='random')
ks = layer.kernel_size(kind='random')
output = layer(x)
export_layer = layer.export()
export_output = export_layer.forward(x)
assert torch.equal(output, export_output)
```
# Mutator
Mutator⽤于调整Mutables ，⽐如应该⽤Mutable中的哪个choice。其⾸先通过解析构建好的⽹络拿到所有的Mutable并将其⾃⾝注册为 Mutable的代理，并实现对应的回调函数。这样Mutable在执⾏forward时，会使⽤Mutator来做真正的操作。常⽤的Mutator有 RandomMutator 、 statelessmutator 等。Mutator有以下属性或方法：

1. model为Mutator的一个属性，类别为nn.Module，model内部存在MixedOp和SliceOp；
2. _parse_mutables()方法，对model进行解析，为model内部所有的Mutables设置同一个Mutator即本身，注意对每一个Mutable只能设置一次Mutator；
3. mutables()方法，返回一个设置mutator为当前Mutator的所有Mutables的生成器；
4. sample_search()方法，遍历设置mutator为当前Muator的所有Mutables，并通过一定算法从Mutable中选择一个候选op进行训练；
5. forward_mixed_op()方法，MixedOp的代理函数，通过调用它实现网络的forward过程；
6. forward_slice_op()方法，SliceOp的代理函数，通过调用它实现网络的forward过程；

以RandomMutator为例给出Mutator的使用例子：
```python
from collections import OrderedDict

import torch
import torch.nn as nn

from gml.models import (BatchNorm2dSlice, Conv2dSlice, MixedOp, RandomMutator,
                        build_mutator)
from gml.space import Categorical, Int

class Net(nn.Module):

    def __init__(self, out_space, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        op_candidates = OrderedDict([('conv3x3', nn.Conv2d(16, 128, 3)),
                                     ('conv5x5', nn.Conv2d(16, 128, 5)),
                                     ('conv7x7', nn.Conv2d(16, 128, 7))])
        self.conv2 = MixedOp(op_candidates, key='mixop')
        op_candidates = OrderedDict([('conv3x3', nn.Conv2d(128, 128, 3)),
                                     ('conv5x5', nn.Conv2d(128, 128, 5)),
                                     ('conv7x7', nn.Conv2d(128, 128, 7))])
        self.conv22 = MixedOp(op_candidates, key='mixop')
        self.out_space = out_space
        self.conv3 = Conv2dSlice(128, out_space, 5)
        self.bn3 = BatchNorm2dSlice(out_space)
        self.act3 = PReLUSlice(out_space)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x

out_space = Int(10, 50, step=20)
model = Net(out_space)
cfg = dict(type='RandomMutator', model=model)
mutator = build_mutator(cfg)
subnet = mutator.sample_search()
x = torch.rand(1, 3, 128, 128)
output = model(x)
```
