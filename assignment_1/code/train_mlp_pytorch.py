"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
nn = torch.nn
optim = torch.optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  accuracy = (predictions.argmax(dim=1) == targets).sum().item()
  accuracy = accuracy / len(predictions)

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Load data
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=False, validation_size=0)
  train_set = cifar10['train']
  test_set = cifar10['test']
  val_set = cifar10['validation']

  # Initialize model
  input_dim = train_set.images[0, :, :, :].size
  n_classes = train_set.labels.max() + 1

  # set device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim, dnn_hidden_units, n_classes)
  model = model.to(device)

  cross_entropy = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=FLAGS.learning_rate, weight_decay=0.002)

  total_loss = 0
  losses = []

  val_acc = []
  train_acc = []
  for i in range(FLAGS.max_steps + 1):

    # prepare batch
    x, y = train_set.next_batch(FLAGS.batch_size)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    x, y = x.to(device), y.to(device)
    x = x.view(FLAGS.batch_size, -1)

    # forward pass
    out = model(x)
    loss = cross_entropy(out, y)
    total_loss += loss.item()

    # keep track of training accuracy
    train_acc.append(accuracy(out, y))

    # backward pass
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if i % FLAGS.eval_freq == 0 and i != 0:

      val_inputs = test_set.images
      val_inputs = torch.tensor(val_inputs)
      val_inputs = val_inputs.to(device)
      val_inputs = val_inputs.view(val_inputs.shape[0], -1)

      pred = model(val_inputs)
      targ = torch.tensor(test_set.labels)
      targ = targ.to(device)

      acc = accuracy(pred, targ)

      losses.append(total_loss)
      val_acc.append(acc)

      print()
      print("- - - - - - - - - -")
      print('- STEPS:\t\t\t', i)
      print('- TRAIN ACC: \t\t\t', np.array(train_acc).mean())
      print('- VALIDATION ACC:\t\t', acc)
      print("- - - - - - - - - -")

      train_acc = []
      total_loss = 0

  print("Loss over time: \t", losses)
  print("Val acc over time: \t", val_acc)

  save_plots('test.png', losses, val_acc)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))


def save_plots(filename, loss, acc):

  examples = np.arange(len(loss)) * FLAGS.eval_freq * FLAGS.batch_size

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('num examples')
  ax1.set_ylabel('loss', color=color)
  ax1.plot(examples, loss, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('val acc', color=color)  # we already handled the x-label with ax1
  ax2.plot(examples, acc, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.savefig(filename=filename)

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
