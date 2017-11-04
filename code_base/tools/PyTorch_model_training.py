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
    return train_input, train_target, valid_input, valid_target, test_input, test_target, data_mean, data_std


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

    train_input, train_target, valid_input, valid_target, test_input, test_target, data_mean, data_std = prepare_data(cf)
    # build the model
    model = LSTM_ManyToMany(input_dim=6, hidden_size=50, num_layers=2, output_size=6)
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
        results = pred[1].data.numpy() * data_std + data_mean
        rect_anno = valid_target.data.numpy() * data_std + data_mean
        aveErrCoverage, aveErrCenter, errCoverage, errCenter = calc_seq_err_robust(results, rect_anno)
        print('aveErrCoverage: %.4f, aveErrCenter: %.2f' % (aveErrCoverage, aveErrCenter))
        print('valid loss:', loss.data.numpy()[0])
        # TODO: 3D evaluation
        # TODO: network saving and evaluation


def calc_seq_err_robust(results, rect_anno):
    """
    :param results:
    :param rect_anno: N*8*6: N is the batch number, 8 frames to predict(seq_length) and 6 is
                    [centreX, centreY, height, width, d_min, d_max]
    :return:
    """

    seq_length = results.shape[1]

    for batch_num in range(len(results)):
        res = results[batch_num]
        anno = rect_anno[batch_num]

        centerGT = [[r[0], r[1], r[4]] for r in anno]
        center = [[r[0], r[1], r[4]] for r in res]

        errCenter = [ssd_2d(center[i], centerGT[i]) for i in range(seq_length)]

        iou_2d = calc_rect_int_2d(res, anno)
        errCoverage = np.zeros(seq_length)
        totalerrCoverage = 0
        totalerrCenter = 0

        for i in range(seq_length):
            errCoverage[i] = iou_2d[i]
            totalerrCoverage += errCoverage[i]
            totalerrCenter += errCenter[i]

        aveErrCoverage = totalerrCoverage / float(seq_length)
        aveErrCenter = totalerrCenter / float(seq_length)

    return aveErrCoverage, aveErrCenter, errCoverage, errCenter

def ssd_2d(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return np.sqrt(s)


def ssd_3d(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return np.sqrt(s)


def calc_rect_int_2d(A, B):
    leftA = [a[0] - a[2] / 2 for a in A]
    bottomA = [a[1] - a[3] / 2 for a in A]
    rightA = [a[0] + a[2]/2 for a in A]
    topA = [a[1] + a[3]/2 for a in A]

    leftB = [b[0] - b[2] / 2 for b in B]
    bottomB = [b[1] - b[3] / 2 for b in B]
    rightB = [b[0] + b[2] / 2 for b in B]
    topB = [b[1] + b[3] / 2 for b in B]

    overlap = []
    length = min(len(leftA), len(leftB))
    for i in range(length):
        tmp = (max(0, min(rightA[i], rightB[i]) - max(leftA[i], leftB[i]) + 1)
               * max(0, min(topA[i], topB[i]) - max(bottomA[i], bottomB[i]) + 1))
        areaA = A[i][2] * A[i][3]
        areaB = B[i][2] * B[i][3]
        overlap.append(tmp / float(areaA + areaB - tmp))

    return overlap