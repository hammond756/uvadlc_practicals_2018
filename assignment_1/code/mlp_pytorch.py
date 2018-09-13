"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn, functional
from functools import reduce

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    network__ = """
    Initializes MLP object.

    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    TODO:
    Implement initialization of the network.
    """

    super(MLP, self).__init__()

    if len(n_hidden) == 0:
      modules = [nn.Linear(n_inputs, n_classes)]
    else:
      modules = [nn.Linear(n_inputs, n_hidden[0])]
      modules.append(nn.ReLU())

      for i in range(1, len(n_hidden)):
        modules.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
        modules.append(nn.ReLU())

      modules.append(nn.Linear(n_hidden[-1], n_classes))

    modules.append(nn.Softmax(dim=1))
    self.net = nn.ModuleList(modules)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    out = reduce(lambda output, module: module(output), self.net, x)

    return out


model = MLP(10, [15, 40, 3], 2)