import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import torch
full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2,
                              12: 3,
                              13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12,
                              26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}

# Calculates class intersections over unions
def iou(pred, target, num_classes):
  ious = []
  # Ignore IoU for background class
  for cls in range(num_classes - 1):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
    union = pred_inds.long().sum() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection / max(union, 1))
  return ious

def calculate_iou(nb_classes, res_dir, label_dir, image_list, label_list):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    # mean_acc = 0.
    for img_num, label_num in zip(image_list, label_list):
        total += 1
        print('#%d: %s' % (total, img_num))
        pred = img_to_array(Image.open('%s/%s' % (res_dir, img_num))).astype(int)
        label = img_to_array(Image.open('%s/%s' % (label_dir, label_num))).astype(int)
        remapped_label = label.copy()
        for k, v in full_to_train.items():
            remapped_label[label == k] = v

        flat_pred = np.ravel(pred)
        flat_label = np.ravel(remapped_label)
        # acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 19:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)

        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print 'mean acc: %f'%mean_acc
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU

def calculate_iou2(nb_classes, res_dir, label_dir, image_list, label_list):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    total_ious = []
    # mean_acc = 0.
    for img_num, label_num in zip(image_list, label_list):
        total += 1
        print('#%d: %s' % (total, img_num))
        pred = Image.open('%s/%s' % (res_dir, img_num))
        label = Image.open('%s/%s' % (label_dir, label_num))
        w, h = pred.size
        pred = torch.ByteTensor(torch.ByteStorage.from_buffer(pred.tobytes())).view(h, w).long()
        label = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes())).view(h, w).long()

        remapped_label = label.clone()
        for k, v in full_to_train.items():
            remapped_label[label == k] = v

        total_ious.append(iou(pred, label, nb_classes))


        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print 'mean acc: %f'%mean_acc
    total_ious = torch.Tensor(total_ious).transpose(0, 1)
    ious = torch.Tensor(19)
    for i, class_iou in enumerate(total_ious):
        ious[i] = class_iou[class_iou == class_iou].mean()  # Calculate mean, ignoring NaNs
    print(ious, ious.mean())
    return ious.mean()

root = '/home/public/CITYSCAPE/'
path = '/home/public/CITYSCAPE/leftImg8bit/test'

val_images = '/home/public/CITYSCAPE/val_images.txt'
val_labels = '/home/public/CITYSCAPE/val_labels.txt'

file = open(val_images, 'r')
file2 = open(val_labels, 'r')
lines = file.readlines()
lines2 = file2.readlines()
file.close()
file2.close()
image_list = []
label_list = []
for line in lines:
    line = line.strip('\n')
    image_list.append(line)

for line in lines2:
    line = line.strip('\n')
    label_list.append(line)

print (image_list)
print (label_list)
print ((len(image_list)))

# for image_num, label_num in zip(image_list, label_list):
#     print (image_num, '-----', label_num)
path = '/home/ty/code/autonomous_driving/Experiments/CityScape_semantic_segmentation'
conf_m, IOU, meanIOU = calculate_iou(19, path, root, image_list, label_list)

# meanIOU = calculate_iou2(19, '/home/ty/code/drn/pred_000/', root, image_list, label_list)

print (meanIOU)

