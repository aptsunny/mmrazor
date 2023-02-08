_base_ = ['./zennas_plainnet_supernet_8xb128_in1k.py']

# Initialized Model Parameters
model = dict(
    type='mmrazor.HYBRIDNAS',
    architecture=_base_.supernet,
    mutator=dict(type='mmrazor.NasMutator'))

# ZeroShotLoop
train_cfg = dict(
    _delete_=True,
    type='mmrazor.ZeroShotLoop',
    max_epochs=480000,
    population_size=512,
    estimator_cfg=dict(type='mmrazor.TrainFreeEstimator', metric='Zen', batch_size=64)
)
