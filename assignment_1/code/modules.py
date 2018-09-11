"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

def exp_normalize_batch(x):
    b = x.max(axis=1)[:, None]
    y = np.exp(x - b)
    return y / y.sum(axis=1)[:, None]

class LinearModule(object):

  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    self.params = {
        'weight': np.random.normal(loc=0, scale=0.0001, size=(in_features, out_features)),
        'bias': np.zeros(out_features)
    }

    self.grads = {
        'weight': np.zeros_like(self.params['weight']),
        'bias': np.zeros_like(self.params['bias'])
    }

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """


    self.prev_x = x

    # Expand dimensions of parameters so we can matmul with batched input
    W = self.params['weight'][None, :]
    b = self.params['bias'][None, :]

    out = np.matmul(x, W) + b

    # TODO: is this needed?
    # remove extra dimension
    out = out.squeeze(0)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

     # TODO: Check this: this way of averaging feels kinda implicit
    self.grads['weight'] = np.matmul(self.prev_x.T, dout)
    self.grads['bias'] = dout.mean(axis=0)

    assert self.grads['weight'].shape == self.params['weight'].shape, "Gradient matrix should be the same shape as params: {}, {}".format(self.grads['weight'].shape, self.params['weight'].shape)

    dx = np.matmul(dout, self.params['weight'].T)
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.prev_x = x
    out = x*(x > 0.0)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    dx = dout*(self.prev_x > 0)

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.S = exp_normalize_batch(x)
    out = self.S

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    dsoft = -np.einsum('ij,ik->ijk', self.S, self.S) # NxD * NxD -> NxDxD
    diag = np.einsum('ij,ik->ijk', self.S, (1 - self.S)).diagonal(axis1=1, axis2=2) # -> NxD (diagonals)

    diag_idx = np.arange(0, dout.shape[1])
    dsoft[:, diag_idx, diag_idx] = diag

    # dx = np.matmul(dout[:, None, :], dsoft).squeeze(1)
    dx = np.einsum('ik,ijk->ij', dout, dsoft)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
