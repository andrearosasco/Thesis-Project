import torch
from torch.optim import SGD

# loss, modelli e dati qui sono inizializzati
# con valori a caso.


def criterion(x, y):
    return (x * y).sum()
lr = 0.1
model_init = torch.tensor([1.,2.,3.], requires_grad=True)
meta_examples = torch.tensor([1.,2.,3.], requires_grad=True)

model_otim = SGD([model_init], lr=0.1)
img_optim = SGD([meta_examples], lr=0.1)

data = torch.tensor([1.,2.,.3])

# calcolo la loss sui meta-esempi
loss = criterion(model_init, meta_examples)
# create_graph è fondamentale per poter fare la backprop
# sul gradiente, altrimenti viene considerato costante.
print(f'model gradient {model_init.grad}')
loss.backward(create_graph=True)

# il nuovo modello è definito esplicitamente
# con un passo di SGD
model_otim.step()


new_loss = criterion(model_init, data)

# meta_examples hanno ancora il gradiente vecchio
print(f'gradiente vecchio {meta_examples.grad}')
# ricordati di azzerrarlo!
meta_examples.grad = None

new_loss.backward()
# Nota che ora il gradiente dei meta-esempi è aggiornato
print(f'gradiente nuovo {meta_examples.grad}')