"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

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

  accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

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

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Load data
  cifar10 = cifar10_utils.get_cifar10(data_dir='cifar10/cifar-10-batches-py', one_hot=True, validation_size=5000)
  train_set = cifar10['train']
  test_set = cifar10['test']
  val_set = cifar10['validation']

  # Initialize model
  input_dim = train_set.images[0, :, :, :].size
  n_classes = train_set.labels[0, :].size

  model = MLP(input_dim, dnn_hidden_units, n_classes)
  cross_entropy = CrossEntropyModule()

  curr_epoch = 0
  train_acc = []
  while train_set.epochs_completed <= FLAGS.max_steps:
    x, y = train_set.next_batch(FLAGS.batch_size)
    x = x.reshape(FLAGS.batch_size, -1)

    out = model.forward(x)
    loss = cross_entropy.forward(out, y)

    train_acc.append(accuracy(out, y))

    # backward pass
    dout = cross_entropy.backward(out, y)
    _ = model.backward(dout)

    # update weights using calculated gradients
    for module in model.modules:
      if type(module) == LinearModule:
        module.params['weight'] = module.params['weight'] - FLAGS.learning_rate * module.grads['weight']
        module.params['bias'] = module.params['bias'] - FLAGS.learning_rate * module.grads['bias']


    if train_set.epochs_completed > curr_epoch:

      val_inputs = val_set.images.reshape(val_set.images.shape[0], -1)

      pred = model.forward(val_inputs)
      targ = val_set.labels

      acc = accuracy(pred, targ)

      print()
      print("- - - - - - - - - -")
      print('- EPOCH:\t\t\t', curr_epoch)
      print('- TRAIN ACC: \t\t\t', np.array(train_acc).mean())
      print('- VALIDATION ACC:\t\t', acc)
      print("- - - - - - - - - -")

      train_acc = []
      curr_epoch = train_set.epochs_completed

      if curr_epoch == 20: break

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