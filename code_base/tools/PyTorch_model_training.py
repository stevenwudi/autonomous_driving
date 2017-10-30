import h5py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from code_base.models.PyTorch_PredictModels import LSTM_ManyToMany


def normalise_data(train_data, valid_data, test_data):
    data_mean = train_data[:, :, :].mean(axis=0).mean(axis=0)
    data_std = train_data[:, :, :].std(axis=0).mean(axis=0)

    train_data -= data_mean
    train_data /= data_std
    valid_data -= data_mean
    valid_data /= data_std
    test_data -= data_mean
    test_data /= data_std
    return train_data, valid_data, test_data, data_mean, data_std


def prepare_data(cf):
    save_dir = os.path.join(cf.shared_path, cf.problem_type)
    f = h5py.File(os.path.join(save_dir, cf.sequence_name + ".hdf5"), "r")
    train_data = f['train_data']
    valid_data = f['valid_data']
    test_data = f['test_data']
    train_data, valid_data, test_data, data_mean, data_std = normalise_data(train_data, valid_data, test_data)
    # train_input = Variable(torch.from_numpy(train_data[:, :cf.lstm_input_frame, :]), requires_grad=False)
    # train_target = Variable(torch.from_numpy(train_data[:, cf.lstm_input_frame:, :]), requires_grad=False)
    valid_input = Variable(torch.from_numpy(train_data[:, :cf.lstm_input_frame, :]), requires_grad=False)
    valid_target = Variable(torch.from_numpy(train_data[:, cf.lstm_input_frame:, :]), requires_grad=False)
    test_input = Variable(torch.from_numpy(train_data[:, :cf.lstm_input_frame, :]), requires_grad=False)
    test_target = Variable(torch.from_numpy(train_data[:, cf.lstm_input_frame:, :]), requires_grad=False)
    # Many to many input
    train_input = Variable(torch.from_numpy(train_data[:, :-1, :]), requires_grad=False)
    train_target = Variable(torch.from_numpy(train_data[:, 1:, :]), requires_grad=False)
    # valid_input = Variable(torch.from_numpy(train_data[:, :-1, :]), requires_grad=False)
    # valid_target = Variable(torch.from_numpy(train_data[:, 1:, :]), requires_grad=False)
    # test_input = Variable(torch.from_numpy(train_data[:, :-1, :]), requires_grad=False)
    # test_target = Variable(torch.from_numpy(train_data[:, 1:, :]), requires_grad=False)
    return train_input, train_target, valid_input, valid_target, test_input, test_target


def get_criterion(cf):
    if cf.loss == 'MSE':
        criterion = nn.MSELoss()
    return criterion


def get_optimisor(cf, model):
    # use LBFGS as optimizer since we can load the whole data to train
    if cf.optimizer == 'LBFGS':
        optimiser = optim.LBFGS(model.parameters(), lr=cf.learning_rate)
    elif cf.optimizer == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=cf.learning_rate)
    elif cf.optimizer == 'rmsprop':
        optimiser = optim.RMSprop(model.parameters(), lr=cf.learning_rate, momentum=cf.momentum,
                                  weight_decay=cf.weight_decay)
    elif cf.optimizer == 'sgd':
        optimiser = optim.SGD(model.parameters(), lr=cf.learning_rate, momentum=cf.momentum,
                              weight_decay=cf.weight_decay)

    return optimiser


def baseline_lstm(cf):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    train_input, train_target, valid_input, valid_target, test_input, test_target = prepare_data(cf)
    # build the model
    model = LSTM_ManyToMany(inputsize=5, hiddensize=50, numlayers=2, outputsize=5)
    model.double()
    if cf.cuda:
        model.cuda()
    criterion = get_criterion(cf)
    optimiser = get_optimisor(cf, model)

    # begin to train
    for e in range(cf.n_epochs):
        print('STEP: ', e)
        def closure():
            optimiser.zero_grad()
            out = model(train_input)[0]
            loss = criterion(out, train_target)
            print('loss: ', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimiser.step(closure)

        # begin to predict
        pred = model(valid_input, future=cf.lstm_predict_frame)
        loss = criterion(pred[1], valid_target)
        print('valid loss:', loss.data.numpy()[0])
