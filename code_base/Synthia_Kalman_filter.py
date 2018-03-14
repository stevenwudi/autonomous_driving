import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from code_base.tools.kalman_filtering import kalman_xy
import seaborn


def get_img_list(valid_data, test_data):

    valid_img_list = []
    for d in valid_data:
        item_list = [x[6] for x in d]
        valid_img_list.append(item_list)

    test_img_list = []
    for d in test_data:
        item_list = [x[6] for x in d]
        test_img_list.append(item_list)

    return  valid_img_list, test_img_list


def get_data_with_img_list(valid_data, test_data):
    valid_data_array = np.array(valid_data[:, :, :6]).astype('float')
    test_data_array = np.array(test_data[:, :, :6]).astype('float')
    return  valid_data_array, test_data_array


def prepare_data_image_list(cf):
    import pickle
    with open(os.path.join(cf['trajectory_path'], cf['sequence_name'] + '_valid.npy'), 'rb') as fp:
        valid_data = pickle.load(fp)
    with open(os.path.join(cf['trajectory_path'], cf['sequence_name'] + '_test.npy'), 'rb') as fp:
        test_data = pickle.load(fp)

    valid_data_array, test_data_array = get_data_with_img_list(valid_data, test_data)
    valid_img_list, test_img_list = get_img_list(valid_data, test_data)

    return  valid_data_array, test_data_array, valid_img_list, test_img_list


def ssd_2d(x, y):
    s = 0
    for i in range(2):
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


def demo_kalman_xy():

    valid_data_array = np.load(r'C:\Users\steve\Desktop\cvpr_figure\car_trajectory_prediction\valid_data_array.npy')
    test_data_array = np.load(r'C:\Users\steve\Desktop\cvpr_figure\car_trajectory_prediction\test_data_array.npy')
    print('Finish Loading')
    valid_pred = np.zeros(shape=(len(valid_data_array), 8, 6))
    valid_pred[:, :, 5:] = valid_data_array[:, -8:, 5:]
    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.

    np.save(os.path.join(cf['trajectory_path'], 'valid_data_array'), valid_data_array)
    np.save(os.path.join(cf['trajectory_path'], 'test_data_array'), test_data_array)
    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4))*1000  # initial uncertainty

    for f, data in enumerate(valid_data_array):
        if f%100 == 0:
            print("process batch %d" %f)
        observed_x = np.array([d[0] for d in data])
        observed_y = np.array([d[1] for d in data])

        result = []
        R = 1**2
        for meas in zip(observed_x[:15], observed_y[:15]):
            x, P = kalman_xy(x, P, meas, R)
            result.append((x[:2]).tolist())
        result_new = []
        result_new.append([result[-1][0], result[-1][1]])
        valid_pred[f, 0, 0] = result[-1][0][0]
        valid_pred[f, 0, 1] = result[-1][1][0]
        for t in range(7):
            x_pred = result_new[-1][0] + x[2]
            y_pred = result_new[-1][1] + x[3]
            valid_pred[f, t + 1, 0] = np.array(x_pred)[0][0]
            valid_pred[f, t + 1, 1] = np.array(y_pred)[0][0]
            valid_pred[f, t + 1, 2] = valid_data_array[f, 14, 2]  # we set the width to be fixed
            valid_pred[f, t + 1, 3] = valid_data_array[f, 14, 3]  # we set the width to be fixed


    totalerrCoverage = 0
    totalerrCenter = 0
    Kalman_valid_errCenter = []
    Kalman_valid_iou_2d = []
    for i in range(len(valid_pred)):
        res = valid_pred[i]
        anno = valid_data_array[i, -8:, :]
        center = [[r[0], r[1]] for r in res]
        centerGT = [[r[0], r[1]] for r in anno]
        seq_length = len(centerGT)
        errCenter = [ssd_2d(center[i], centerGT[i]) for i in range(len(centerGT))]
        Kalman_valid_errCenter.append(errCenter)
        iou_2d = calc_rect_int_2d(res, anno)
        Kalman_valid_iou_2d.append(iou_2d)
        for s in range(seq_length):
            totalerrCenter += errCenter[s]
            totalerrCoverage += iou_2d[s]

    aveErrCoverage = totalerrCoverage / (len(Kalman_valid_errCenter) * float(seq_length))
    aveErrCenter = totalerrCenter /  (len(Kalman_valid_errCenter) * float(seq_length))
    print('aveErrCoverage: %0.3f, aveErrCenter: %0.3f ' %(aveErrCoverage, aveErrCenter))
    # aveErrCoverage: 0.547, aveErrCenter: 28.128
    np.save(r'C:\Users\steve\Desktop\Figures\tracking_plot\Kalman_valid_errCenter', Kalman_valid_errCenter)
    np.save(r'C:\Users\steve\Desktop\Figures\tracking_plot\KKalman_valid_iou_2d', Kalman_valid_iou_2d)
    Kalman_valid_errCenter = []
    Kalman_valid_iou_2d = []



if __name__ == "__main__":
    # Entry point of the script
    demo_kalman_xy()