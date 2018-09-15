"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
nn = torch.nn
optim = torch.optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  # Load data
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=False, validation_size=5000)
  train_set = cifar10['train']
  test_set = cifar10['test']
  val_set = cifar10['validation']

  # Initialize model
  n_channels = len(train_set.images[0].shape)
  n_classes = train_set.labels.max() + 1

  # set device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = ConvNet(n_channels, n_classes)
  model = model.to(device)

  cross_entropy = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(),lr=FLAGS.learning_rate)

  total_loss = 0
  losses = []

  val_acc = []
  train_acc = []
  for i in range(FLAGS.max_steps + 1):

    # prepare batch
    x, y = train_set.next_batch(FLAGS.batch_size)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.long)
    x, y = x.to(device), y.to(device)

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

      val_inputs = val_set.images
      val_inputs = torch.tensor(val_inputs)
      val_inputs = val_inputs.to(device)

      pred = model(val_inputs)
      targ = torch.tensor(val_set.labels)
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

  # save_plots('test.png', losses, val_acc)


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

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
