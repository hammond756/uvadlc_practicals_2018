################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
nn = torch.nn
optim = torch.optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

from pycrayon import CrayonClient

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def one_hot(batch, vocab_size):

    X = torch.zeros(batch.shape[0], batch.shape[1], vocab_size, device=batch.device)
    X.scatter_(2, batch[:,:,None], 1)

    return X

def get_predictions(outputs):
    return torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(seq_length=config.input_length,
                           input_dim=config.input_dim,
                           num_hidden=config.num_hidden,
                           num_classes=config.num_classes,
                           batch_size=config.batch_size)
    elif config.model_type == 'LSTM':
        model = LSTM(seq_length=config.input_length,
                           input_dim=config.input_dim,
                           num_hidden=config.num_hidden,
                           num_classes=config.num_classes,
                           batch_size=config.batch_size)

    experiment_label = "{}_".format(datetime.now().strftime("%Y-%m-%d %H:%M"))
    for key, value in vars(config).items():
        experiment_label += "{}={}_".format(key, value)

    # TensorBoard API
    cc = CrayonClient(hostname='18.202.229.41', port=6007)
    xp = cc.create_experiment(xp_name=experiment_label)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        if config.input_dim == 10:
            batch_inputs = one_hot(batch_inputs.type(torch.long), config.input_dim)

        p = model(batch_inputs)

        loss = criterion(p, batch_targets)
        predictions = get_predictions(p)
        accuracy = (predictions == batch_targets).sum().item() / len(predictions)

        # backward pass
        model.zero_grad()
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            # post metrics to tensorboard
            xp.add_scalar_dict({
                'accuracy' : accuracy,
                'loss' : loss.item()
            }, wall_time=time.time(), step=step)

        if step == config.train_steps:

            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    _ = xp.to_zip(experiment_label + ".zip")
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)