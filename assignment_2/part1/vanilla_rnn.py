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

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        # Initial distribution of weights
        mean = 0.0
        std = 0.001

        # Initialize parameters
        self.Wxh = nn.Parameter(nn.init.normal_(torch.empty(input_dim, num_hidden), mean=mean, std=std))
        self.Whh = nn.Parameter(nn.init.normal_(torch.empty(num_hidden, num_hidden), mean=mean, std=std))
        self.bh = nn.Parameter(torch.zeros(num_hidden))
        self.Whp = nn.Parameter(nn.init.normal_(torch.empty(num_hidden, num_classes), mean=mean, std=std))
        self.bp = nn.Parameter(torch.zeros(num_classes))

        # Save meta information
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.input_dim = input_dim

    def forward(self, x):

        self.h = torch.zeros(self.batch_size, self.num_hidden)

        for t in range(self.seq_length):

            # convert input at time (t) to [BxD] vector
            x_t = x[:, t].view(self.batch_size, self.input_dim)

            # compute hidden state
            self.h = torch.mm(x_t, self.Wxh) + torch.mm(self.h, self.Whh) + self.bh
            self.h = torch.tanh(self.h)


        return torch.mm(self.h, self.Whp) + self.bp
