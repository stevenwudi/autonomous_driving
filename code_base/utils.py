import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils
import torch


def HMS(sec):
    '''
    :param sec: seconds
    :return: print of H:M:S
    '''

    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)

    return "%dh:%02dm:%02ds" % (h, m, s)


def configurationPATH(cf, dataset_path):
    '''
    :param cf: config file
    :param dataset_path: path where the datased is located
    :return: Print some paths
    '''

    print("\n###########################")
    print(' > Conf File Path = "%s"' % (cf.config_path))
    print(' > Save Path = "%s"' % (cf.savepath))
    print(' > Dataset PATH = "%s"' % (os.path.join(cf.dataroot_dir)))
    print("###########################\n")


# Sets the backend and GPU device.
class Environment():
    def __init__(self, backend='tensorflow'):
        backend = 'tensorflow'  # 'theano' or 'tensorflow'
        os.environ['KERAS_BACKEND'] = backend
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # "" to run in CPU, extra slow! just for debuging
        if backend == 'theano':
            # os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'
            """ fast_compile que lo que hace es desactivar las optimizaciones => mas lento """
            os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.95'
            print('Backend is Theano now')
        else:
            print('Backend is Tensorflow now')


def show_DG(DG, show_set='train'):
    # Let's instantiate this class the iterate through the data samples.
    train_set = DG.dataloader[show_set]

    for i_batch, sample_batched in enumerate(train_set):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched, DG)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


# Helper function to show a batch
def show_landmarks_batch(sample_batched, DG):
    """Show image with landmarks for a batch of samples."""
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['label']

    grid_images = utils.make_grid(images_batch, nrow=5, padding=10)
    # because the utils.make_grid requires tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
    # also the grid image is of dtype DoubleTensor
    labels_batch = labels_batch.unsqueeze(1).type(torch.DoubleTensor)
    grid_labels = utils.make_grid(labels_batch, nrow=5, padding=10, normalize=True)
    # we need to expand the labels range
    plt.subplot(2, 1, 1)
    plt.imshow(grid_images.numpy().transpose(1, 2, 0))
    plt.subplot(2, 1, 2)
    plt.imshow(grid_labels.numpy().transpose(1, 2, 0)[:,:,0], cmap='jet')
    plt.title('Batch from dataloader')