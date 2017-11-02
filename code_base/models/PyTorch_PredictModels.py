from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM_ManyToMany(nn.Module):
    """
    This model like characterlevel prediction or time_sequence_prediction(https://github.com/pytorch/examples/tree/master/time_sequence_prediction):
    During train stage & test stage, it receive one input and output next prediction;
    During perdict stage, it sample current output as next input, iteratively.
    So, it is very like the caption model in CS231n, but its input has no image features.
    It has 2 hidden layers, and the output layer is a linear layer.
    """
    def __init__(self, inputsize, hiddensize, numlayers, outputsize):
        super(LSTM_ManyToMany, self).__init__()
        # some superParameters & input output dimensions
        self.input_size = inputsize
        self.hiddensize = hiddensize
        self.num_layers = numlayers
        self.output_size = outputsize
        # build lstm layer, parameters are (input_size,hiddensize,num_layers)
        self.lstm = nn.LSTM(self.input_size, self.hiddensize, self.num_layers, batch_first=True)
        # build output layer, which is a linear layer
        self.linear = nn.Linear(self.hiddensize, self.output_size)

    def forward(self, input, future=0):
        # the return content
        outputs = []
        # init hidden state & cell state
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hiddensize).double(), requires_grad=False)
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hiddensize).double(), requires_grad=False)

        # compute
        outputs_lstm, (hn, cn) = self.lstm(input, (h_0, c_0))
        outputs_linear = self.linear(outputs_lstm)

        # result of train or test, shape(batch size,sequence size, feature size)
        outputs += [outputs_linear]

        # if we should predict the future
        if future > 0:
            h_t = hn
            c_t = cn
            output_linear = outputs_linear[:, -1, :]  # the last output during test
            size = output_linear.size()
            output_linear = output_linear.resize(size[0], 1, size[1])
            outputs_pre = []
            for i in range(future):
                output_lstm, (h_t, c_t) = self.lstm(output_linear, (h_t, c_t))
                output_linear = self.linear(output_lstm)
                outputs_pre += [output_linear.squeeze(1)]
            outputs_pre = torch.stack(outputs_pre, 1)
            outputs += [outputs_pre]

        return outputs