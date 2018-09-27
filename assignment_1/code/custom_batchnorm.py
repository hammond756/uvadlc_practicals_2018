import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones((n_neurons,)))
    self.beta = nn.Parameter(torch.zeros((n_neurons,)))

    # self.gamma = nn.Parameter(torch.Tensor((n_neurons,)))
    # self.beta = nn.Parameter(torch.Tensor((n_neurons,)))

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    assert input.shape[1] == self.n_neurons, "input should be of size {}, got {}".format(self.n_neurons, input.size())

    mu = input.mean(dim=0)
    var = input.var(dim=0, unbiased=False)
    x_hat = (input - mu) / torch.sqrt(var + self.eps)

    out = torch.mul(self.gamma, x_hat) + self.beta

    assert input.shape == out.shape, "input shape shouldn't change: {} -> {}".format(input.shape, out.shape)

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrieval of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shape (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """


    ctx.eps = eps

    # calculate numerator
    x = input
    mu = x.mean(dim=0)
    x_mu_diff = x - mu

    # calculate denomenator
    square_diff = x_mu_diff.pow(2)
    var = square_diff.mean(dim=0)
    sig = torch.sqrt(var + eps)
    sig_inv = 1 / sig

    # put them together
    x_hat = torch.mul(x_mu_diff, sig_inv)

    # scale
    scaled = torch.mul(gamma, x_hat)

    # shift
    shifted = scaled + beta

    out = shifted

    ctx.save_for_backward(gamma, mu, x_mu_diff, var, sig, sig_inv, x_hat)

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    N, D = grad_output.shape

    gamma, mu, x_mu_diff, var, sig, sig_inv, x_hat = ctx.saved_tensors

    # shifting
    grad_gamma_x = grad_output

    if ctx.needs_input_grad[2]:
      grad_beta = grad_output.sum(dim=0)
    else:
      grad_beta = None

    # scaling
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
      grad_gamma = torch.sum(torch.mul(grad_gamma_x, x_hat), dim=0)
    else:
      grad_gamma = None

    if ctx.needs_input_grad[0]:
      grad_x_hat = torch.mul(grad_gamma_x, gamma)

      assert grad_x_hat.shape == (N,D)

      # normalization
      grad_sig_inv = torch.sum(torch.mul(grad_x_hat, x_mu_diff), dim=0)
      grad_x_mu_diff_1 = torch.mul(grad_x_hat, sig_inv)

      assert grad_x_mu_diff_1.shape == (N,D)

      # inverting standard deviation
      grad_sig = torch.mul(-1 / sig.pow(2), grad_sig_inv)

      assert grad_sig.shape == (D,)

      # rooting the variance
      grad_var = torch.mul((0.5 / sig), grad_sig) # sig = sqrt(var + eps)

      # averaging the square differences
      grad_pow2 = torch.mul(1 / N * torch.ones((N,D), dtype=torch.double), grad_var)

      # sqaure-ing the difference
      grad_x_mu_diff_2 = 2 * torch.mul(x_mu_diff, grad_pow2)

      # substracting the mean
      grad_input_1 = grad_x_mu_diff_1 + grad_x_mu_diff_2
      grad_mu = -1 * torch.sum((grad_x_mu_diff_1 + grad_x_mu_diff_2), dim=0)
      # averaging the batch
      grad_input_2 = torch.mul(1 / N * torch.ones((N,D), dtype=torch.double), grad_mu)

      # finally the input
      grad_input = grad_input_1 + grad_input_2
    else:
      grad_input = None

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones((n_neurons,)))
    self.beta = nn.Parameter(torch.zeros((n_neurons,)))

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    assert input.shape[1] == self.n_neurons

    bn_func = CustomBatchNormManualFunction(self.n_neurons)
    out = bn_func.apply(input, self.gamma, self.beta)

    return out