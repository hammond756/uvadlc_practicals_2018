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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        # Initial distribution weights
        mean = 0.0
        std = 0.001

        # Helper variable
        concat_len = input_dim + num_hidden

        # Initialize linear operators
        self.W_f = nn.Parameter(nn.init.normal_(torch.empty(concat_len, num_hidden), mean=mean, std=std))
        self.b_f = nn.Parameter(torch.zeros(num_hidden))

        self.W_i = nn.Parameter(nn.init.normal_(torch.empty(concat_len, num_hidden), mean=mean, std=std))
        self.b_i = nn.Parameter(torch.zeros(num_hidden))

        self.W_C = nn.Parameter(nn.init.normal_(torch.empty(concat_len, num_hidden), mean=mean, std=std))
        self.b_C = nn.Parameter(torch.zeros(num_hidden))

        self.W_o = nn.Parameter(nn.init.normal_(torch.empty(concat_len, num_hidden), mean=mean, std=std))
        self.b_o = nn.Parameter(torch.zeros(num_classes))

        # Added this myself to map h to num_classes
        self.W_p = nn.Parameter(nn.init.normal_(torch.empty(num_hidden, num_classes), mean=mean, std=std))
        self.b_p = nn.Parameter(torch.zeros(num_classes))

        # Save meta information
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size

    def forward(self, x):

        # initialize cell state
        C = torch.zeros(self.batch_size, self.num_hidden)
        h = torch.zeros(self.batch_size, self.num_hidden)

        for t in range(self.seq_length):
            x_t = x[:,t].view(self.batch_size, self.input_dim)

            concat = torch.cat([h, x_t], dim=1)

            # forget gate
            f = torch.mm(concat, self.W_f) + self.b_f
            f = torch.sigmoid(f)

            # input gate
            i = torch.mm(concat, self.W_i) + self.b_i
            i = torch.sigmoid(i)

            # candidate cell state
            C_ = torch.mm(concat, self.W_C) + self.b_C
            C_ = torch.tanh(C_)

            # compute new cell state
            C = torch.mul(f, C) + torch.mul(i, C_)

            # output gate
            o = torch.mm(concat, self.W_o)
            o = torch.sigmoid(o)

            # sort term memory
            h = torch.mul(o, torch.tanh(C))

        return torch.mm(h, self.W_p) + self.b_p