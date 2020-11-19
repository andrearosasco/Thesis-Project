lr_list = list(range(10))
lrs = {f'Learning rate {i}': lr for (i, lr) in enumerate(lr_list)}
metrics = {f'Distill train loss': 5, f'Distill train accuracy': 5, f'Distill step': 6}
print(type({**lrs, **metrics}))