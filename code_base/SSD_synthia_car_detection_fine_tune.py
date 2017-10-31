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

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from code_base.models.Keras_SSD import SSD512v2, BBoxUtility, Generator, MultiboxLoss
priors = pickle.load(
    open('/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/models/prior_boxes_ssd512.pkl', 'rb'),
    encoding='latin1')
bbox_util = BBoxUtility(NUM_CLASSES, priors)


def combine_gt(annotation_1, annotation_2, merged_annotation):
    annotations_list_1 = json.load(open(annotation_1, 'r'))
    annotations_list_2 = json.load(open(annotation_2, 'r'))
    with open(merged_annotation, 'w') as outfile:
        json.dump(annotations_list_1+annotations_list_2, outfile)


def converting_gt(annotations_url, gt_file):

    gt = defaultdict(list)

    def f_annotation(l):
        gt = {}
        count = 0
        for el in l:
            img_path = el['image_path'] +'/' + el['image_name']
            gt[img_path] = []
            for anno in el['boundingbox']:
                count += 1
                gt_annot = np.zeros(4+NUM_CLASSES-1)
                # anno[0] = min(max(0, anno[0]), resize_train[0])
                # anno[1] = min(max(0, anno[1]), resize_train[1])
                xmin = anno[0] / resize_train[1]
                ymin = anno[1] / resize_train[0]
                xmax = anno[2] / resize_train[1]
                ymax = anno[3] / resize_train[0]
                gt_annot[:4] = [xmin, ymin, xmax, ymax]
                gt_annot[4] = 1
                gt[img_path].append(gt_annot)
        print('Finish converting, total annotated fish number is %d in total image of %d.'%(count, len(gt)))
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


def train_ssd512(gt_file):

    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.summary()
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

    base_lr = 1e-4
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

    nb_epoch = 50
    history = model.fit_generator(generator=gen.generate(True),
                                  steps_per_epoch=gen.train_batches,
                                  epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  workers=1)


def test_ssd512(gt_file, model_checkpoint):

    gt = pickle.load(open(gt_file, 'rb'))
    gt = gt_classification_convert(gt)
    keys = sorted(gt.keys())
    random.shuffle(keys)

    num_train = int(round(0.9 * len(keys)))
    val_keys = keys[num_train:]

    ### load model ###
    model = SSD512v2(input_shape, num_classes=NUM_CLASSES)
    model.load_weights(model_checkpoint, by_name=True)

    inputs = []
    images = []
    add_num = 40 + 19
    gt_result = []
    # for i in range(num_val):
    for i in range(20):
        img_path = val_keys[i + add_num]
        if os.path.isfile(img_path):
            gt_result.append(gt[val_keys[i + add_num]])
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
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            # Get detections with confidence higher than 0.6.
            # top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]
            top_indices = [0]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
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
        if len(gt_img):
            gt_top_xmin = gt_img[0][0]
            gt_top_ymin = gt_img[0][1]
            gt_top_xmax = gt_img[0][2]
            gt_top_ymax = gt_img[0][3]

            xmin = int(round(gt_top_xmin * img.shape[1]))
            ymin = int(round(gt_top_ymin * img.shape[0]))
            xmax = int(round(gt_top_xmax * img.shape[1]))
            ymax = int(round(gt_top_ymax * img.shape[0]))
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = 'r'
            ## gt label
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))

        plt.draw()
        plt.waitforbuttonpress(3)


def main():

    merged_annotation = '/home/public/synthia/ssd_car_fine_tune/SYNTHIA-SEQS-01-TRAIN_MERGED.json'
    if False:
        # we combine the training and validation here
        annotations_url_1 = '/home/public/synthia/SYNTHIA-SEQS-01-TRAIN.json'
        annotations_url_2 = '/home/public/synthia/SYNTHIA-SEQS-01-VALIDATE.json'
        combine_gt(annotations_url_1, annotations_url_2, merged_annotation)

    gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_fine_tune_gt.pkl'
    if False:
        # for training annotation conversion
        converting_gt(merged_annotation, gt_file)
    if False:
        # Training
        train_ssd512(gt_file)

    test_gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test_gt.pkl'
    if False:
        # converting testing GT
        annotations_url = '/home/public/synthia/SYNTHIA-SEQS-01-TEST.json'
        converting_gt(annotations_url, test_gt_file)
    if True:
        # Testing
        model_checkpoint = '/home/public/synthia/ssd_car_fine_tune/weights_512.01-0.45.hdf5'
        test_ssd512(test_gt_file, model_checkpoint)


if __name__ == "__main__":
    main()
