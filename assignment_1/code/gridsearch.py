__author__ = 'Aron'

from mlp_pytorch import MLP
import cifar10_utils
import torch
nn = torch.nn
import skorch
from sklearn.model_selection import GridSearchCV
import pickle

def main():

  # Load data
  cifar10 = cifar10_utils.get_cifar10(one_hot=False, validation_size=0)
  train_set = cifar10['train']

  # Initialize model
  input_dim = train_set.images[0, :, :, :].size
  n_classes = train_set.labels.max() + 1

  net = skorch.NeuralNetClassifier(
    MLP,
    criterion=nn.CrossEntropyLoss,
    module__n_inputs=input_dim,
    module__n_classes=n_classes,
    optimizer=torch.optim.SGD
  )

  params = {
    'lr' : [0.1, 0.01, 0.001],
    'module__n_hidden' : [
      [100, 100],
      [1000],
      [50,30,20]
    ],
    'batch_size' : [64, 128, 256, 512]
  }
  # params = {
  #   'lr' : [0.1, 0.01],
  #   'module__n_hidden' : [[100]]
  # }

  gs = GridSearchCV(net, params, cv=5, scoring='accuracy', n_jobs=4, refit=False, verbose=2)
  gs.fit(train_set.images.reshape(train_set.images.shape[0], -1), train_set.labels)

  print()
  print('--------')
  print()
  print('Best params:\t', gs.best_params_)
  print('Best score:\t', gs.best_score_)
  print()

  with open('gridsearch_full_training_set.pickle', 'wb') as f:
    pickle.dump(gs, f)


if __name__ == '__main__':

  main()