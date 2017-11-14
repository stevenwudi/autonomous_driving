import h5py
import os
import numpy as np
import cv2 as cv
from scipy.misc import imresize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def normalise_data(train_data, valid_data, test_data):
    data_mean = train_data[:, :, :].mean(axis=0).mean(axis=0)
    data_std = train_data[:, :, :].std(axis=0).mean(axis=0)
    valid_data_mean = valid_data[:, :, :].mean(axis=0).mean(axis=0)
    valid_data_std = valid_data[:, :, :].std(axis=0).mean(axis=0)
    test_data_mean = test_data[:, :, :].mean(axis=0).mean(axis=0)
    test_data_std = test_data[:, :, :].std(axis=0).mean(axis=0)

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

    #--------> to test data shuffle
    np.random.seed(10)
    train_size = train_data.shape[0]
    valid_size = valid_data.shape[0]

    all_data = np.concatenate((train_data, valid_data, test_data), axis=0)
    all_data = all_data.astype(float)
    np.random.shuffle(all_data)
    # train_data = all_data[:train_size, :, :]
    train_data = all_data[:512, :, :]
    valid_data = all_data[train_size:train_size+valid_size, :, :]
    test_data = all_data[train_size+valid_size:, :, :]
    # --------< to test data shuffle

    train_data, valid_data, test_data, data_mean, data_std = normalise_data(train_data,
                                                                            valid_data,
                                                                            test_data)

    if cf.cuda:
        print('Data using CUDA')
        dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU
    else:
        dtype = torch.FloatTensor

    valid_input = Variable(torch.from_numpy(valid_data[:, :cf.lstm_input_frame, :]).type(dtype), requires_grad=False)
    valid_target = Variable(torch.from_numpy(valid_data[:, cf.lstm_input_frame:, :]).type(dtype), requires_grad=False)
    test_input = Variable(torch.from_numpy(test_data[:, :cf.lstm_input_frame, :]).type(dtype), requires_grad=False)
    test_target = Variable(torch.from_numpy(test_data[:, cf.lstm_input_frame:, :]).type(dtype), requires_grad=False)
    # Many to many input
    if cf.model_name == 'LSTM_ManyToMany':
        train_input = Variable(torch.from_numpy(train_data[:, :-1, :]).type(dtype), requires_grad=False)
        train_target = Variable(torch.from_numpy(train_data[:, 1:, :]).type(dtype), requires_grad=False)
    else:
        train_input = Variable(torch.from_numpy(train_data[:, :cf.lstm_input_frame, :]).type(dtype),
                                requires_grad=False)
        train_target = Variable(torch.from_numpy(train_data[:, cf.lstm_input_frame:, :]).type(dtype),
                                requires_grad=False)


    # images: the input images, may be semantic segmentation or RGB. size as (batchSize, sequenceSize, Cin, Hin, Win)
    train_images = Variable(torch.zeros(train_input.size(0), train_input.size(1), 1, 10, 10).type(dtype), requires_grad=False)
    valid_images = Variable(torch.zeros(valid_input.size(0), valid_input.size(1), 1, 10, 10).type(dtype), requires_grad=False)
    test_images = Variable(torch.zeros(test_input.size(0), test_input.size(1), 1, 10, 10).type(dtype), requires_grad=False)

    return train_images, valid_images, test_images, train_input, train_target, valid_input, valid_target, test_input, test_target, data_mean, data_std


def normalise_data_with_img_list(train_data, valid_data, test_data):
    train_data_array = np.array(train_data[:, :, :6]).astype('float')
    valid_data_array = np.array(valid_data[:, :, :6]).astype('float')
    test_data_array = np.array(test_data[:, :, :6]).astype('float')

    data_mean = train_data_array[:, :, :].mean(axis=0).mean(axis=0)
    data_std = train_data_array[:, :, :].std(axis=0).mean(axis=0)

    train_data_array -= data_mean
    train_data_array /= data_std
    valid_data_array -= data_mean
    valid_data_array /= data_std
    test_data_array -= data_mean
    test_data_array /= data_std
    return train_data_array, valid_data_array, test_data_array, data_mean, data_std


def get_img_list(train_data, valid_data, test_data):
    train_img_list = []
    for d in train_data:
        item_list = [x[6] for x in d]
        train_img_list.append(item_list)

    valid_img_list = []
    for d in valid_data:
        item_list = [x[6] for x in d]
        valid_img_list.append(item_list)

    test_img_list = []
    for d in test_data:
        item_list = [x[6] for x in d]
        test_img_list.append(item_list)

    return train_img_list, valid_img_list, test_img_list

def get_img_resized_list(cf, train_data, valid_data, test_data):

    root_dir = '/'.join(cf.dataset_path[0].split('/')[:-1])
    # def data_img_dir(data):
    #     return root_dir + '/' + data[0][0][6].split('/')[
    #         0] + '/' + 'GT/LABELS' + '/' + cf.data_stereo + '/' + cf.data_camera
    #
    # train_img_dir = data_img_dir(train_data)
    # valid_img_dir = data_img_dir(valid_data)
    # test_img_dir = data_img_dir(test_data)

    def img_name(item):
        return os.path.join(root_dir + '/' + item.split('/')[0] + '/' + 'GT/LABELS' + '/' + cf.data_stereo + '/' + cf.data_camera, item.split('/')[1])

    def semantic_image(img_name):
        try:
            input = cv.imread(img_name, -1)
            semantic_image = np.int8(input[:, :, 2])
        except IOError:
            # unfortunately, some images are corrupted. Hence, we need to manually exclude them.
            print("Image failed loading: ", img_name)
        # resize
        semantic_image = imresize(semantic_image, size=0.125, interp='nearest', mode='F')
        # Convert to training labels
        w, h = semantic_image.shape
        # Create one-hot encoding
        semantic_image_one_hot = np.zeros(shape=(2, w, h))
        car_chanel = 8
        road_chanel = 3
        for c, chanel in enumerate([car_chanel, road_chanel]):
            semantic_image_one_hot[c][semantic_image == chanel] = 1
        # Convert to tensors
        semantic_image_t = torch.Tensor(semantic_image_one_hot)
        return semantic_image_t

    save_root_dir = '/media/samsumg_1tb/synthia/SYNTHIA-SEQS-01/resized_semantic_images/'
    def save_semantics_itemlist(name, itemlist):
        path = save_root_dir + name + '.npy'
        np.save(path, [itemlist])
        return path

    valid_img_list = []
    i = 0
    for d in valid_data:
        i += 1
        item_list = [x[6] for x in d[:cf.lstm_input_frame]]
        item_list = [semantic_image(img_name(item)) for item in item_list]
        item_list = torch.stack(item_list, dim=0)
        valid_img_list.append(save_semantics_itemlist('valid_'+str(i), item_list))

    test_img_list = []
    i = 0
    for d in test_data:
        i += 1
        item_list = [x[6] for x in d[:cf.lstm_input_frame]]
        item_list = [semantic_image(img_name(item)) for item in item_list]
        item_list = torch.stack(item_list, dim=0)
        test_img_list.append(save_semantics_itemlist('test_' + str(i), item_list))

    train_img_list = []
    i = 0
    for d in train_data:
        i += 1
        item_list = [x[6] for x in d[:cf.lstm_input_frame]]
        item_list = [semantic_image(img_name(item)) for item in item_list]
        item_list = torch.stack(item_list, dim=0)
        train_img_list.append(save_semantics_itemlist('train_' + str(i), item_list))

    return train_img_list, valid_img_list, test_img_list

#
def prepare_data_image_list(cf):
    if cf.dataloader_load_prepare_data:
        # load data
        prepared_data = tuple(np.load(cf.dataloader_load_prepare_data_path))
    else:
        import pickle
        with open(os.path.join(cf.shared_path, cf.problem_type, cf.sequence_name + '_train.npy'), 'rb') as fp:
            train_data = pickle.load(fp)
        with open(os.path.join(cf.shared_path, cf.problem_type, cf.sequence_name + '_valid.npy'), 'rb') as fp:
            valid_data = pickle.load(fp)
        with open(os.path.join(cf.shared_path, cf.problem_type, cf.sequence_name + '_test.npy'), 'rb') as fp:
            test_data = pickle.load(fp)

        train_data_array, valid_data_array, test_data_array, data_mean, data_std = normalise_data_with_img_list(train_data,
                                                                                                                valid_data,
                                                                                                                test_data)
        train_img_list, valid_img_list, test_img_list = get_img_resized_list(cf, train_data, valid_data, test_data)

        prepared_data = (train_data_array, valid_data_array, test_data_array, data_mean, data_std, train_img_list, valid_img_list, test_img_list)

        if cf.dataloader_save_prepare_data:
            # save data
            np.save(cf.dataloader_save_prepare_data_path, prepared_data)

    return prepared_data


def calc_seq_err_robust(results, rect_anno, focal_length):
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

        centerGT = [[r[0], r[1]] for r in anno]
        center = [[r[0], r[1]] for r in res]

        errCenter = [ssd_2d(center[i], centerGT[i]) for i in range(seq_length)]

        centerGT_realworld = [[r[0]/focal_length*r[4], r[1]/focal_length*r[4], r[4], r[5]] for r in anno/100.]
        center_realworld = [[r[0]/focal_length*r[4], r[1]/focal_length*r[4], r[4], r[5]] for r in res/100.]
        errCenter_realworld = [ssd_3d(center_realworld[i], centerGT_realworld[i]) for i in range(seq_length)]

        iou_2d = calc_rect_int_2d(res, anno)
        iou_3d = calc_rect_int_3d(res, anno, focal_length)
        errCoverage = np.zeros(seq_length)
        totalerrCoverage = 0
        totalerrCenter = 0
        totalerrCoverage_realworld = 0
        totalerrCenter_realworld = 0
        for i in range(seq_length):
            totalerrCenter += errCenter[i]
            totalerrCoverage += iou_2d[i]
            totalerrCenter_realworld += errCenter_realworld[i]
            totalerrCoverage_realworld += iou_3d[i]

        aveErrCoverage = totalerrCoverage / float(seq_length)
        aveErrCenter = totalerrCenter / float(seq_length)
        aveErrCoverage_realworld = totalerrCoverage_realworld / float(seq_length)
        aveErrCenter_realworld = totalerrCenter_realworld / float(seq_length)

    return aveErrCoverage, aveErrCenter, errCenter, iou_2d, \
           aveErrCoverage_realworld, aveErrCenter_realworld, errCenter_realworld, iou_3d


def ssd_2d(x, y):
    s = 0
    for i in range(2):
        s += (x[i] - y[i]) ** 2
    return np.sqrt(s)


def ssd_3d(x, y):
    s = 0
    for i in range(3):
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


def calc_rect_int_3d(A, B, focal_length):

    proj_A_All = [a[4]/focal_length for a in A]
    proj_B_All = [b[4] / focal_length for b in B]

    leftA = [proj_A*(a[0] - a[2] / 2) for (proj_A, a) in zip(proj_A_All, A)]
    bottomA = [proj_A*(a[1] - a[3] / 2) for (proj_A, a) in zip(proj_A_All, A)]
    rightA = [proj_A*(a[0] + a[2]/2) for (proj_A, a) in zip(proj_A_All, A)]
    topA = [proj_A*(a[1] + a[3]/2) for (proj_A, a) in zip(proj_A_All, A)]
    closest_A = [a[4] for a in A]
    farthest_A = [a[5] for a in A]

    leftB = [proj_B*(b[0] - b[2] / 2) for (proj_B, b) in zip(proj_B_All, B)]
    bottomB = [proj_B*(b[1] - b[3] / 2) for (proj_B, b) in zip(proj_B_All, B)]
    rightB = [proj_B*(b[0] + b[2] / 2) for (proj_B, b) in zip(proj_B_All, B)]
    topB = [proj_B*(b[1] + b[3] / 2) for (proj_B, b) in zip(proj_B_All, B)]
    closest_B = [b[4] for b in B]
    farthest_B = [b[5] for b in B]

    overlap = []
    length = min(len(leftA), len(leftB))
    for i in range(length):
        tmp = (max(0, min(rightA[i], rightB[i]) - max(leftA[i], leftB[i])) *
               max(0, min(topA[i], topB[i]) - max(bottomA[i], bottomB[i])) *
               max(0, min(farthest_A[i], farthest_B[i]) - max(closest_A[i], closest_B[i])))
        volumn_A = (proj_A_All[i] * A[i][2]) * (proj_A_All[i] * A[i][3]) * (farthest_A[i] - closest_A[i])
        volumn_B = (proj_B_All[i] * B[i][2]) * (proj_B_All[i] * B[i][3]) * (farthest_B[i] - closest_B[i])
        overlap.append(tmp / float(volumn_A + volumn_B - tmp))

    return overlap