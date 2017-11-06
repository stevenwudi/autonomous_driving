"""
This script is used to convert car gt to the format of:
[xmin, ymin, xmax, ymax, prob1, prob2, prob3, ...],
xmin, ymin, xmax, ymax are in relative coordinates.
Since car is the only class with one-hot encoding
"""
from collections import defaultdict
import json
import numpy as np
import pickle
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import matplotlib.pyplot as plt

# random seed, it's very important for the experiment...
random.seed(1000)
NUM_CLASSES = 1 + 1
resize_train = (760, 1280)
input_shape = (512, 512, 3)

# import keras
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
# from code_base.models.Keras_SSD import SSD512v2, BBoxUtility, Generator, MultiboxLoss
# priors = pickle.load(
#     open('/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/models/prior_boxes_ssd512.pkl', 'rb'),
#     encoding='latin1')
# bbox_util = BBoxUtility(NUM_CLASSES, priors)


def combine_gt(annotation_1, annotation_2, merged_annotation):
    annotations_list_1 = json.load(open(annotation_1, 'r'))
    annotations_list_2 = json.load(open(annotation_2, 'r'))
    with open(merged_annotation, 'w') as outfile:
        json.dump(annotations_list_1+annotations_list_2, outfile)


def converting_gt(annotations_url, gt_file, POR=None):

    gt = defaultdict(list)

    def f_annotation(l):
        gt = {}
        count = 0
        for el in l:
            img_path = el['image_path'] +'/' + el['image_name']
            gt[img_path] = []
            for anno in el['boundingbox']:
                gt_annot = np.zeros(4+NUM_CLASSES-1)
                xmin = anno[0] / resize_train[1]
                ymin = anno[1] / resize_train[0]
                xmax = anno[2] / resize_train[1]
                ymax = anno[3] / resize_train[0]
                if POR and (ymax-ymin)*(xmax-xmin) < POR:
                    continue
                gt_annot[:4] = [xmin, ymin, xmax, ymax]
                gt_annot[4] = 1
                gt[img_path].append(gt_annot)
                count += 1
        print('Finish converting, total annotated car number is %d in total image of %d.'%(count, len(gt)))
        return gt

    sloth_annotations_list = json.load(open(annotations_url, 'r'))
    gt.update(f_annotation(sloth_annotations_list))

    with open(gt_file, 'wb') as fp:
        pickle.dump(gt, fp)
    print("Finish loading images, total number of images is: " + str(len(gt)))


def gt_classification_convert(gt):
    for key in gt.keys():
        if len(gt[key]) != 0:
            anno_list = []
            for l in gt[key]:
                anno_list.append(l)
            gt[key] = np.asarray(anno_list)
        else:
            gt[key] = np.ndarray(shape=(0, 4 + NUM_CLASSES - 1))

    return gt


def train_ssd512(gt_file, model_checkpoint=None, base_lr=1e-4):

    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.summary()
    if model_checkpoint:
        model.load_weights(model_checkpoint, by_name=True)
    else:
        model.load_weights('/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/models/weights_SSD300.hdf5', by_name=True)
    gt = pickle.load(open(gt_file, 'rb'), encoding='latin1')
    gt = gt_classification_convert(gt)

    keys = sorted(gt.keys())
    random.shuffle(keys)

    num_train = int(round(0.9 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]

    gen = Generator(gt=gt, bbox_util=bbox_util, batch_size=1, path_prefix='',
                    train_keys=train_keys, val_keys=val_keys,
                    image_size=(input_shape[0], input_shape[1]), do_crop=False)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2']
    #       'conv3_1', 'conv3_2', 'conv3_3', 'pool3']  #,
    #       'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    def schedule(epoch, decay=0.9):
        return base_lr * decay ** (epoch)

    callbacks = [keras.callbacks.ModelCheckpoint('/home/public/synthia/ssd_car_fine_tune/weights_512.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    base_lr = base_lr
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

    nb_epoch = 100
    history = model.fit_generator(generator=gen.generate(True),
                                  steps_per_epoch=gen.train_batches,
                                  epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  workers=1)


def examine_ssd512(gt_file, model_checkpoint):

    gt = pickle.load(open(gt_file, 'rb'))
    gt = gt_classification_convert(gt)
    keys = sorted(gt.keys())
    random.shuffle(keys)
    ### load model ###
    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights(model_checkpoint, by_name=True)

    inputs = []
    images = []
    add_num = 0
    gt_result = []
    # for i in range(num_val):
    for i in range(20):
        img_path = keys[i + add_num]
        if os.path.isfile(img_path):
            gt_result.append(gt[keys[i + add_num]])

            img = image.load_img(img_path, target_size=(512, 512))
            img = image.img_to_array(img)
            images.append(img)
            inputs.append(img.copy())

    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    for i, img in enumerate(images):
        currentAxis = plt.gca()
        currentAxis.cla()
        plt.imshow(img / 255.)
        # Parse the outputs.
        if len(results[i]):
            # det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            for j in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[j] * img.shape[1]))
                ymin = int(round(top_ymin[j] * img.shape[0]))
                xmax = int(round(top_xmax[j] * img.shape[1]))
                ymax = int(round(top_ymax[j] * img.shape[0]))
                score = top_conf[j]
                display_txt = '{:0.2f}'.format(score)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = 'g'
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        # plt GT
        gt_img = gt_result[i]
        for g_num in range(len(gt_img)):
            gt_top_xmin = gt_img[g_num][0]
            gt_top_ymin = gt_img[g_num][1]
            gt_top_xmax = gt_img[g_num][2]
            gt_top_ymax = gt_img[g_num][3]

            xmin = int(round(gt_top_xmin * img.shape[1]))
            ymin = int(round(gt_top_ymin * img.shape[0]))
            xmax = int(round(gt_top_xmax * img.shape[1]))
            ymax = int(round(gt_top_ymax * img.shape[0]))
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = 'r'
            ## gt label
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))

        plt.draw()
        plt.waitforbuttonpress(3)


def test_ssd512(gt_file, model_checkpoint, test_json_file):
    # mAP_threshold=np.linspace(0.5, 0.95, num=10)
    gt = pickle.load(open(gt_file, 'rb'))
    gt = gt_classification_convert(gt)
    keys = sorted(gt.keys())
    random.shuffle(keys)
    ### load model ###
    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights(model_checkpoint, by_name=True)

    predict_dict = {}

    for i in range(len(keys)):
        img_path = keys[i]
        print(img_path)
        if os.path.isfile(img_path):
            img = image.load_img(img_path, target_size=(512, 512))
            img = image.img_to_array(img)
            # we process image frame by frame
            inputs = preprocess_input(np.array([img]))
            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)
            detected_rect = []
            if len(results[0]):
                # TODO: check which image has no detection
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                # Get detections with confidence higher than 0.6.
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
                top_conf = det_conf[top_indices]
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for j in range(len(top_indices)):
                    detected_rect.append([top_xmin[j], top_ymin[j], top_xmax[j], top_ymax[j], top_conf[j]])
            predict_dict[img_path] = detected_rect

    with open(test_json_file, 'w') as fp:
        json.dump(predict_dict, fp, indent=4)


def calculate_iou(test_gt_file, test_json_file, POR=None, draw=False):
    """

    :param test_gt_file:
    :param test_json_file:
    :param POR: the pixel occupant rate, if smaller than this value, we ignore it both
                for gt and prediction
    :return:
    """
    # loading predicted json file
    from code_base.tools.yolo_utils import box_iou, BoundBox
    with open(test_json_file, 'r') as fp:
        predict_dict = json.load(fp)

    gt = pickle.load(open(test_gt_file, 'rb'))
    gt_dict = gt_classification_convert(gt)

    conf_threshold = np.linspace(0.5, 0.95, num=10)
    mAP_threshold = np.linspace(0.5, 0.95, num=10)
    tp = np.zeros(shape=(len(conf_threshold), len(mAP_threshold)))
    total_pred = np.zeros(len(conf_threshold))
    total_true = 0
    for i, k in enumerate(gt_dict.keys()):
        boxes_true = []
        boxes_pred = []

        for b in predict_dict[k]:
            bx = BoundBox(1)
            bx.x, bx.y,  bx.c = b[0], b[1], b[-1]
            bx.w, bx.h = b[2]-b[0], b[3]-b[1]
            boxes_pred.append(bx)

        for g in gt_dict[k]:
            gx = BoundBox(1)
            gx.x, gx.y, gx.c = g[0], g[1], g[-1]
            gx.w, gx.h = g[2]-g[0], g[3]-g[1]
            if POR and gx.w * gx.h < POR:
                continue
            # we count all GT boxes
            boxes_true.append(gx)
            total_true += 1

        for c, detection_threshold in enumerate(conf_threshold):
            true_matched_pred = np.zeros(shape=(len(boxes_pred), len(mAP_threshold)))
            for ib, bx in enumerate(boxes_pred):
                if POR and bx.w * bx.h < POR:
                    continue
                if bx.c < detection_threshold:
                    continue
                total_pred[c] += 1
                true_matched = np.zeros(shape=(len(boxes_true), len(mAP_threshold)))
                for u, iou_threshold in enumerate(mAP_threshold):
                    for t, gx in enumerate(boxes_true):
                        if true_matched[t, u]:
                            continue
                        if box_iou(gx, bx) > iou_threshold:
                            true_matched[t, u] = 1
                            true_matched_pred[ib, u] = 1
                            tp[c, u] += 1.
                            break

        if draw:
            # we only draw conf=0.5, mAP(oou)=0.5) with false postive and false negative
            if len(true_matched_pred) != np.sum(true_matched_pred[:,0]) or len(true_matched) != np.sum(true_matched[:,0]):
                img = plt.imread(k)
                currentAxis = plt.gca()
                currentAxis.cla()
                plt.imshow(img)
                # first we draw false positive as green
                fp = true_matched_pred[:, 0] == 0
                for idx_fp, fp_v in enumerate(fp):
                    if fp_v:
                        bp = boxes_pred[idx_fp]
                        x, y, w, h, conf = bp.x, bp.y, bp.w, bp.h, bp.c
                        x *= img.shape[1]
                        y *= img.shape[0]
                        w *= img.shape[1]
                        h *= img.shape[0]
                        coords = (x, y), w, h
                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=1))
                        currentAxis.text(x, y, '%.2f' % (conf), color='g')
                # Then we plot the false negative as red
                if len(boxes_true):
                    fn = true_matched[:, 0] == 0
                    for idx_fn, fn_v in enumerate(fn):
                        if fn_v:
                            bt = boxes_true[idx_fn]
                            x, y, w, h, conf = bt.x, bt.y, bt.w, bt.h, bt.c
                            x *= img.shape[1]
                            y *= img.shape[0]
                            w *= img.shape[1]
                            h *= img.shape[0]
                            coords = (x, y), w, h
                            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='r', linewidth=1))
                            #currentAxis.text(xmin, ymin, 'FN', bbox={'facecolor': 'r', 'alpha': 0.5})
            plt.draw()
            plt.waitforbuttonpress(3)

    precision = tp / total_pred
    recall = tp / total_true
    f = np.divide(2 * np.multiply(precision, recall),  (precision + recall))

    np.set_printoptions(precision=3)
    print('Conf: %s' % (np.array_str(conf_threshold)))
    print('Total GT: %d. \n Total prediction: %s' % (total_true, np.array_str(total_pred)))
    print('Precision: %s' % (np.array_str(precision[:, 0])))
    print('Recall: %s' % (np.array_str(recall[:, 0])))
    print('F score: %s' % (np.array_str(f)))


def collect_front_and_rear_gt(annotation_1, annotation_2, save_dir, image_interval=50):
    from PIL import Image
    import scipy.misc
    annotations_list_1 = json.load(open(annotation_1, 'r'))
    annotations_list_2 = json.load(open(annotation_2, 'r'))
    print(len(annotations_list_1), len(annotations_list_2))
    car_count = 0
    for i in range(0, len(annotations_list_1), image_interval):
        img = Image.open(annotations_list_1[i]['image_path']+'/'+annotations_list_1[i]['image_name'])
        img = np.array(img)
        for bb_idx, bb in enumerate(annotations_list_1[i]['boundingbox']):
            img_car = img[bb[1]:bb[3], bb[0]:bb[2]]
            img_name = str(car_count) + '.png'
            print(img_name)
            car_count += 1
            scipy.misc.imsave(os.path.join(save_dir, img_name), img_car)

    for i in range(0, len(annotations_list_2), image_interval):
        img = Image.open(annotations_list_2[i]['image_path']+'/'+annotations_list_2[i]['image_name'])
        img = np.array(img)
        for bb_idx, bb in enumerate(annotations_list_2[i]['boundingbox']):
            img_car = img[bb[1]:bb[3], bb[0]:bb[2]]
            img_name = str(car_count) + '.png'
            print(img_name)
            car_count += 1
            scipy.misc.imsave(os.path.join(save_dir, img_name), img_car)


def ssd_synthia_car_fine_tune():
    """
    The scirpt to calling different modules for fine-tuning/verifying SSD
    :return:
    """
    merged_annotation = '/home/public/synthia/ssd_car_fine_tune/SYNTHIA-SEQS-01-TRAIN_MERGED-shuffle.json'
    if False:
        print('we combine the training and validation here')
        annotations_url_1 = '/home/public/synthia/SYNTHIA-SEQS-01-TRAIN-shuffle.json'
        annotations_url_2 = '/home/public/synthia/SYNTHIA-SEQS-01-VALIDATE-shuffle.json'
        combine_gt(annotations_url_1, annotations_url_2, merged_annotation)

    if True:
        print('collect front and rear cars')
        annotations_url_1 = '/home/public/synthia/SYNTHIA-SEQS-01-TRAIN-shuffle.json'
        annotations_url_2 = '/home/public/synthia/SYNTHIA-SEQS-01-VALIDATE-shuffle.json'
        save_dir = '/home/stevenwudi/PycharmProjects/autonomous_driving/Experiments/SEQ_01_SEQ_06_cars'
        collect_front_and_rear_gt(annotations_url_1, annotations_url_2, save_dir, image_interval=50)

    gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_fine_tune_gt-shuffle.pkl'
    if False:
        print('Training annotation conversion')
        converting_gt(merged_annotation, gt_file, POR=1e-3)
        # POR: 1e-3  Finish converting, total annotated car number is 22332 in total image of 8814.
        # POR: 5e-4: Finish converting, total annotated fish number is 26800 in total image of 8814.

    model_checkpoint = '/home/public/synthia/ssd_car_fine_tune/weights_512.54-0.19.hdf5'
    if False:
        print('Start DDS 512 training')
        train_ssd512(gt_file, model_checkpoint=model_checkpoint, base_lr=1e-5)

    test_gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test_gt-shuffle.pkl'
    if False:
        print('Converting testing GT')
        annotations_url = '/home/public/synthia/SYNTHIA-SEQS-01-TEST-shuffle.json'
        converting_gt(annotations_url, test_gt_file)
    if False:
        # Examine test data
        examine_ssd512(test_gt_file, model_checkpoint)

    test_json_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test-shuffle.json'
    if False:
        test_ssd512(test_gt_file, model_checkpoint, test_json_file)
    # A separate file for accepting gt file and predicted json fil
    if False:
        calculate_iou(test_gt_file, test_json_file, POR=2e-3, draw=True)
    """
    This is the network results train by SSD512 (with 0.05% POR trained)
    Conf: [ 0.5   0.55  0.6   0.65  0.7   0.75  0.8   0.85  0.9   0.95]
    
    ### POR=1e-3
    Total GT: 2327. 
    Total prediction: [ 2016.  1986.  1951.  1918.  1888.  1852.  1815.  1774.  1711.  1641.]
    Precision: [ 0.913  0.908  0.9    0.895  0.886  0.879  0.867  0.853  0.83   0.804]
    Recall: [ 0.791  0.787  0.78   0.775  0.768  0.761  0.751  0.739  0.719  0.697]
    F score: [[ 0.847  0.843  0.837  0.825  0.816  0.79   0.748  0.666  0.449  0.106]
     [ 0.843  0.839  0.834  0.823  0.813  0.788  0.748  0.665  0.449  0.106]
     [ 0.835  0.833  0.829  0.818  0.809  0.786  0.747  0.665  0.449  0.106]
     [ 0.831  0.829  0.826  0.815  0.807  0.784  0.746  0.664  0.449  0.105]
     [ 0.823  0.822  0.82   0.81   0.802  0.781  0.744  0.663  0.447  0.105]
     [ 0.816  0.816  0.815  0.806  0.799  0.778  0.741  0.661  0.446  0.105]
     [ 0.805  0.805  0.806  0.797  0.792  0.772  0.738  0.66   0.445  0.105]
     [ 0.792  0.793  0.793  0.789  0.784  0.766  0.734  0.658  0.445  0.105]
     [ 0.77   0.773  0.775  0.771  0.768  0.753  0.725  0.652  0.441  0.104]
     [ 0.746  0.749  0.751  0.751  0.75   0.741  0.716  0.645  0.439  0.103]]
         
    ### POR shuffle
    Total GT: 2563. 
    Total prediction: [ 2185.  2137.  2081.  2047.  2010.  1976.  1937.  1893.  1831.  1713.]
    Precision: [ 0.843  0.836  0.826  0.819  0.813  0.805  0.795  0.784  0.766  0.726]
    Recall: [ 0.719  0.712  0.704  0.698  0.693  0.686  0.678  0.669  0.653  0.619]
    F score: [[ 0.776  0.698  0.646  0.619  0.597  0.571  0.525  0.447  0.31   0.106]
     [ 0.769  0.693  0.642  0.616  0.595  0.57   0.524  0.447  0.31   0.106]
     [ 0.76   0.686  0.637  0.612  0.591  0.568  0.523  0.445  0.31   0.106]
     [ 0.754  0.681  0.632  0.608  0.588  0.565  0.521  0.443  0.31   0.106]
     [ 0.749  0.677  0.629  0.605  0.586  0.563  0.52   0.442  0.309  0.106]
     [ 0.741  0.671  0.624  0.602  0.583  0.561  0.518  0.441  0.309  0.106]
     [ 0.732  0.664  0.619  0.597  0.579  0.558  0.516  0.44   0.308  0.106]
     [ 0.722  0.656  0.612  0.59   0.573  0.553  0.515  0.439  0.308  0.106]
     [ 0.705  0.642  0.6    0.58   0.564  0.545  0.508  0.435  0.306  0.105]
     [ 0.668  0.614  0.578  0.561  0.548  0.531  0.496  0.426  0.302  0.105]]
    
    POR = 2e-3
    Total GT: 1433. 
    Total prediction: [ 1438.  1428.  1421.  1413.  1403.  1398.  1384.  1374.  1348.  1305.]
    Precision: [ 0.928  0.926  0.924  0.922  0.919  0.917  0.91   0.906  0.898  0.876]
    Recall: [ 0.931  0.929  0.927  0.925  0.922  0.92   0.913  0.909  0.901  0.879]
    F score: [[ 0.929  0.921  0.91   0.901  0.883  0.851  0.785  0.673  0.476  0.164]
     [ 0.927  0.92   0.908  0.9    0.882  0.851  0.785  0.673  0.476  0.164]
     [ 0.926  0.919  0.907  0.899  0.882  0.851  0.785  0.673  0.476  0.164]
     [ 0.924  0.916  0.905  0.897  0.88   0.849  0.785  0.673  0.476  0.164]
     [ 0.92   0.914  0.903  0.895  0.878  0.847  0.783  0.671  0.475  0.164]
     [ 0.918  0.912  0.901  0.894  0.877  0.846  0.782  0.671  0.475  0.164]
     [ 0.912  0.907  0.898  0.89   0.874  0.843  0.781  0.67   0.475  0.164]
     [ 0.908  0.903  0.895  0.888  0.871  0.84   0.78   0.67   0.475  0.164]
     [ 0.899  0.896  0.888  0.882  0.866  0.836  0.776  0.668  0.474  0.164]
     [ 0.878  0.877  0.872  0.869  0.855  0.829  0.77   0.663  0.471  0.164]]
 
 
     ### POR = None (consider all testing examples)
    Total GT: 2696. 
    Total prediction: [ 2273.  2221.  2166.  2111.  2055.  2000.  1945.  1875.  1786.  1684.]
    Precision: [ 0.822  0.818  0.81   0.805  0.796  0.789  0.777  0.763  0.741  0.716]
    Recall: [ 0.693  0.69   0.683  0.678  0.671  0.665  0.655  0.643  0.625  0.604]
    F score: [[ 0.752  0.746  0.74   0.729  0.724  0.703  0.668  0.597  0.405  0.096]
     [ 0.748  0.744  0.737  0.727  0.722  0.702  0.668  0.597  0.405  0.096]
     [ 0.741  0.737  0.733  0.723  0.718  0.699  0.667  0.596  0.405  0.096]
     [ 0.736  0.733  0.729  0.72   0.716  0.698  0.666  0.596  0.404  0.095]
     [ 0.728  0.726  0.724  0.716  0.712  0.695  0.664  0.595  0.403  0.095]
     [ 0.722  0.721  0.72   0.712  0.708  0.692  0.661  0.593  0.402  0.095]
     [ 0.711  0.711  0.711  0.704  0.703  0.687  0.659  0.592  0.401  0.095]
     [ 0.698  0.7    0.7    0.696  0.696  0.682  0.655  0.59   0.401  0.095]
     [ 0.678  0.682  0.684  0.681  0.681  0.67   0.647  0.585  0.398  0.095]
     [ 0.655  0.659  0.662  0.663  0.665  0.659  0.639  0.578  0.395  0.094]]
 """


if __name__ == "__main__":
    ssd_synthia_car_fine_tune()
