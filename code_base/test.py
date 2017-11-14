import os
from skimage import io
path = '/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets/segmentation/SYNTHIA_RAND_CVPR16/RGB'
imgs = os.listdir(path)
errors = ['ap_000_02-11-2015_18-02-19_000062_3_Rand_2.png',
                'ap_000_02-11-2015_18-02-19_000129_2_Rand_16.png',
                'ap_000_01-11-2015_19-20-57_000008_1_Rand_0.png']

for name in imgs:
    # n, s = os.path.splitext(name)
    if name in ['ap_000_02-11-2015_18-02-19_000062_3_Rand_2.png',
                'ap_000_02-11-2015_18-02-19_000129_2_Rand_16.png',
                'ap_000_01-11-2015_19-20-57_000008_1_Rand_0.png']:
        continue
    else:
        print (name)
        tmp = io.imread(path + '/' + name)