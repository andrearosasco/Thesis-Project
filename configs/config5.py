conf = {
    'exp': 'exp_permuted_mnist',
    'model': 'mlp1',
    'epochs': 2, # best .89 accuracy
    'mb': 256,
    'lr': 0.0003,
    'valid': 0.2,
    'buffer': 100,
    'tasks': range(5),
    'tile': (1, 1)
}