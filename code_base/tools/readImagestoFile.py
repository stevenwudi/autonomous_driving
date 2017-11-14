import os
from matplotlib import pyplot as plt
from PIL import Image
path = '/home/public/CITYSCAPE/leftImg8bit/train'

# img = Image.open('/home/public/CITYSCAPE/gtFine/test/munich/')
folders = os.listdir(path)
image_file = open('/home/public/CITYSCAPE/train_images.txt', 'w')
label_file = open('/home/public/CITYSCAPE/train_labels.txt', 'w')
print (folders)

for folder in folders:
    imgs = os.listdir(path + '/' + folder)
    # print (imgs)
    for img in imgs:
        print (img)
        print (img[:-15])
        image_file.writelines('leftImg8bit/train/' + folder + '/' + img + '\n')
        label_file.writelines('gtFine/train/' + folder + '/' + img[:-15] + 'gtFine_labelIds.png\n')