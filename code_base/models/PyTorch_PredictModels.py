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
    def __init__(self, input_dim, hidden_size, num_layers, output_size, cuda=True):
        super(LSTM_ManyToMany, self).__init__()
        # some superParameters & input output dimensions
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # build lstm layer, parameters are (input_dim,hidden_size,num_layers)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, batch_first=True)
        # build output layer, which is a linear layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, input, future=0):
        """
           input: The train data or test data, type is Variable, size is (batchSize, sequenceSize,featureSize)
           future: The number of predicting frames, it is equivalent to the number of iterative samples
           return: A list composed of 2 elements. The first element is the test output, type is Variable, Size is same as input, (batchSize, sequenceSize,featureSize).
                   The second element is the predicted data, type is Variable, Size is (batchSize, futureSequenceSize,featureSize)
        """
        # the return content
        outputs = []
        # init hidden state & cell state
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).type(self.dtype), requires_grad=False)
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).type(self.dtype), requires_grad=False)

        # compute
        outputs_lstm, (hn, cn) = self.lstm(input, (h_0, c_0))
        outputs_linear = self.linear(outputs_lstm)

        # result of train or test, shape(batch size,sequence size, feature size)
        outputs += [outputs_linear]

        outputs_pre = [outputs_linear[:, -1, :]]
        # if we should predict the future
        if future-1 > 0:
            h_t = hn
            c_t = cn
            output_linear = outputs_linear[:, -1, :]  # the last output during test
            size = output_linear.size()
            output_linear = output_linear.resize(size[0], 1, size[1])
            for i in range(future):
                output_lstm, (h_t, c_t) = self.lstm(output_linear, (h_t, c_t))
                output_linear = self.linear(output_lstm)
                outputs_pre += [output_linear.squeeze(1)]

        outputs_pre = torch.stack(outputs_pre, 1)
        outputs += [outputs_pre]

        return outputs


class LSTM_To_FC(nn.Module):
    """
    This model use an LSTM to fuse features of all past frames into a single state, which inputs to a following Fully Connected model to predict next some frames' features.
    The lstm model can be seen as a encoder, while the FC model as a decoder.
    """
    def __init__(self, future, input_dim, hidden_size, num_layers, output_dim, cuda=True):
        """
        future: The number of predicting frames
        input_dim: The number of features in each frame
        hidden_size: The number of features in the hidden state of lstm
        num_layers: The number of recurrent layers in lstm
        self.output_size: The number of fc's outputs
        self.linear1_output_size: The number of cells hidden layer of fc model
        """
        super(LSTM_To_FC, self).__init__()
        self.future = future
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_size = self.future * self.output_dim
        self.linear1_output_size = int((self.hidden_size + self.output_size)/2)

        # build lstm layer, parameters are (input_dim,hidden_size,num_layers)
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, batch_first=True)
        # build fc model
        self.linear1 = nn.Linear(self.hidden_size, self.linear1_output_size)
        self.nonlinear1 = nn.ReLU()
        self.linear2 = nn.Linear(self.linear1_output_size, self.output_size)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, input, future=0):
        """
        input: The train data or test data, type is Variable, size is (batchSize, sequenceSize,featureSize)
        future: The number of predicting frames, but this parameter is invalid, it is determined by _init_
        return: A list composed of 2 elements. The first element is the input data without any change, to do this is for plot.
                The second element is the predicted data, type is Variable, Size is (batchSize, futureSequenceSize,featureSize)
        """
        # the return content
        outputs = [input]
        # init hidden state & cell state,  parameters are (numlayers, batchsize, hidden_size)
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).type(self.dtype), requires_grad=False)
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).type(self.dtype), requires_grad=False)

        # compute with train or test data,the output is hidden_state & cell_state
        # the hn1[-1] represents hidden_state in last lstm layer, and it involves all the past information through the recurrent process
        outputs_lstm1, (hn1, cn1) = self.lstm1(input, (h_0, c_0))

        # compute the forward in FC model
        output_linear1 = self.linear1(hn1[-1])
        output_nonlinear1 = self.nonlinear1(output_linear1)
        output_linear2 = self.linear2(output_nonlinear1)

        # resize to (batchsize, frame_num, featureNumInEachFrame)
        size = output_linear2.size()
        output_linear2 = output_linear2.resize(size[0], self.future, self.input_dim)
        outputs.append(output_linear2)

        return outputs
