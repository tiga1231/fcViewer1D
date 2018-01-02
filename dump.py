import numpy as np
import matplotlib.pyplot as plt

import json
from time import time

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class MyNet(nn.Module):
    def __init__(self, layers, activation=nn.ReLU()):
        super(MyNet, self).__init__()
        j=0
        for i in range(1,len(layers)):
            self.add_module(str(j), nn.Linear(layers[i-1], layers[i]))
            j+=1
            if activation is not None and i!= len(layers)-1:
                self.add_module(str(j),activation)
                j+=1
            
    def forward(self, x):
        self.activations = []
        self.activations.append(x.clone().data.numpy())
        for module in self._modules.values():
            x = module(x)
            self.activations.append(x.clone().data.numpy())
        return x
    
    
def printNet(net):
    acts = net.activations
    modules = list(net.modules())[1:]
    for layer in range(len(acts)):
        print layer
        print acts[layer].size()
        if layer != len(acts)-1:
            print getattr(net,str(layer))
            


def f(x):
    return torch.sin(x)


neuronCounts = [1,20,10,5,1]
net = MyNet(neuronCounts,nn.LeakyReLU(negative_slope=0.1))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

#training
data = []
n = 10000 # training iterations
pauses = 20 # number of pauses to dump intermediate (activation) data
for step in range(n):
    
    x = Variable(torch.rand(32,neuronCounts[0])*2*3-3)
    y_ = f(x)
    y = net(x)
    
    loss = criterion(y, y_)

    net.zero_grad()
    loss.backward()
    optimizer.step()

    if step%(n/pauses)==0:
        print step, loss.data.numpy()[0]

        x = Variable(torch.Tensor(np.linspace(-3,3,100).reshape([-1,1])))
        y_ = f(x)
        y = net(x)
        loss = float(criterion(y, y_).data.numpy()[0])

        layouts = []
        for a in net.activations:
            if a.shape[1] > 1:
                # model = TSNE(n_components=2, perplexity = 5)
                model = PCA(n_components=2)
                layout = model.fit_transform(a.T)
     
                layout = layout.tolist()
            else:
                layout = 'null'
            layouts.append(layout)

        activations = [a.tolist() for a in net.activations]

        d = {
            'step': step,
            'activations': activations,
            'y_': y_.data.numpy().ravel().tolist(),
            'weights':'null',
            'layouts': layouts,
            'loss': loss,
            'layerNames': ['Input'] + [repr(getattr(net, str(i))) for i in range(len(net.activations)-1)]
        }
        data.append(d)
    


#with open('data/data'+str(int(time()))+'.js', 'w') as f:
with open('data.js', 'w') as f:
    f.write('var data = \n')
    json.dump(data, f)






















