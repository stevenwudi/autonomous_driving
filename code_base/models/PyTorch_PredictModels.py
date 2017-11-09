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
    def __init__(self, input_dims, hidden_sizes, outlayer_input_dim, outlayer_output_dim, cuda=True):
        '''
        :param input_dims: a list involves each lstm_layer's input_dim
        :param hidden_sizes: a list involves each lstm_layer's hidden_size
        :param outlayer_input_dim:
        :param outlayer_output_dim:
        :param cuda: whether or not to use cuda
        '''
        super(LSTM_ManyToMany, self).__init__()
        # some superParameters & input output dimensions
        self.input_dims = input_dims
        self.hidden_sizes = hidden_sizes
        self.outlayer_input_dim = outlayer_input_dim
        self.outlayer_output_dim = outlayer_output_dim

        # build lstm layer, parameters are (input_dim,hidden_size,num_layers)
        for i, input_dim in enumerate(self.input_dims):
            self.__setattr__('lstm' + str(i), nn.LSTM(input_size=input_dim, hidden_size=self.hidden_sizes[i], num_layers=1, batch_first=True))

        # build output layer, which is a linear layer
        self.linear = nn.Linear(in_features=self.outlayer_input_dim, out_features=self.outlayer_output_dim)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, input, future=0):
        '''
        :param input: The train data or test data, type is Variable, size is (batchSize, sequenceSize,featureSize)
        :param future: The number of predicting frames, it is equivalent to the number of iterative samples
        :return: A list composed of 2 elements. The first element is the test output, type is Variable, Size is same as input, (batchSize, sequenceSize,featureSize).
                   The second element is the predicted data, type is Variable, Size is (batchSize, futureSequenceSize,featureSize)
        '''

        # the return content
        outputs = []
        # batch size, for init hidden state & cell state
        batch_size = input.size(0)

        # hidden state & cell state for t=seq_len, size as h0 or c0
        hn = {}
        cn = {}
        # compute
        input_t = input
        for i in range(0, len(self.hidden_sizes)):
            # init hidden state & cell state
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_sizes[i]).type(self.dtype), requires_grad=False)
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_sizes[i]).type(self.dtype), requires_grad=False)
            lstm = self.__getattr__('lstm'+str(i))
            output_t, (hn[i], cn[i]) = lstm(input_t, (h0, c0))
            input_t = output_t
        outputs_linear = self.linear(input_t)

        # result of train or test, shape(batch size,sequence size, feature size)
        outputs += [outputs_linear]

        # result of predict, shape as (batch size,future, feature size)
        outputs_pre = [outputs_linear[:, -1, :]]
        # if we should predict the future
        if future-1 > 0:
            ht = hn
            ct = cn
            output_linear = outputs_linear[:, -1, :]  # the last output during test
            size = output_linear.size()
            output_linear = output_linear.resize(size[0], 1, size[1])
            for j in range(future-1):
                input_t = output_linear
                for i in range(0, len(self.hidden_sizes)):
                    lstm = self.__getattr__('lstm' + str(i))
                    output_t, (ht[i], ct[i]) = lstm(input_t, (ht[i], ct[i]))
                    input_t = output_t
                output_linear = self.linear(input_t)
                outputs_pre += [output_linear.squeeze(1)]
        outputs_pre = torch.stack(outputs_pre, 1)

        outputs += [outputs_pre]

        return outputs


class LSTM_To_FC(nn.Module):
    """
    This model use an LSTM to fuse features of all past frames into a single state, which inputs to a following Fully Connected model to predict next some frames' features.
    The lstm model can be seen as a encoder, while the FC model as a decoder.
    """
    def __init__(self,  input_dims, hidden_sizes, future_frame, output_dim, cuda=True):
        '''

        :param input_dims: a list involves each lstm_layer's input_dim
        :param hidden_sizes: a list involves each lstm_layer's hidden_size
        :param future_frame: the number of predicting frames
        :param output_dim: the number of features in each predicting frame
        :param cuda: whether or not to use GPU
        '''
        super(LSTM_To_FC, self).__init__()
        # lstm part
        self.input_dims = input_dims
        self.hidden_sizes = hidden_sizes
        # fully connected part
        self.future_frame = future_frame
        self.output_dim = output_dim
        self.output_size = self.future_frame * self.output_dim
        self.fc_hidden1_size = int((self.hidden_sizes[-1] + self.output_size)/2)

        # build lstm layer, parameters are (input_dim,hidden_size,num_layers)
        for i, input_dim in enumerate(self.input_dims):
            self.__setattr__('lstm' + str(i),
                             nn.LSTM(input_size=input_dim, hidden_size=self.hidden_sizes[i], num_layers=1,
                                     batch_first=True))

        # build fc model
        self.linear1 = nn.Linear(self.hidden_sizes[-1], self.fc_hidden1_size)
        self.nonlinear1 = nn.ReLU()
        self.linear2 = nn.Linear(self.fc_hidden1_size, self.output_size)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, input):
        """
        input: The train data or test data, type is Variable, size is (batchSize, sequenceSize,featureSize)
        future: The number of predicting frames, but this parameter is invalid, it is determined by _init_
        return: A list composed of only one element, which is the predicted data, type is Variable, Size is (batchSize, futureSequenceSize, featureSize)
        """
        # the return content
        outputs = []

        # compute with train or test data,the output is hidden_state & cell_state
        # the hn represents hidden_state in each lstm layer, and it involves all the past information through the recurrent process
        # compute
        batch_size = input.size(0)
        input_t = input
        for i in range(0, len(self.hidden_sizes)):
            # init hidden state & cell state, parameters are (numlayers, batchsize, hidden_size)
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_sizes[i]).type(self.dtype), requires_grad=False)
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_sizes[i]).type(self.dtype), requires_grad=False)
            lstm = self.__getattr__('lstm'+str(i))
            output_t, (hn, cn) = lstm(input_t, (h0, c0))
            input_t = output_t

        # compute the forward in FC model
        output_linear1 = self.linear1(hn[-1, :, :])
        output_nonlinear1 = self.nonlinear1(output_linear1)
        output_linear2 = self.linear2(output_nonlinear1)

        # resize to (batchsize, frame_num, featureNumInEachFrame)
        size = output_linear2.size()
        output_linear2 = output_linear2.resize(size[0], self.future_frame, self.output_dim)
        outputs.append(output_linear2)

        return outputs
