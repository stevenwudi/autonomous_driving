import os
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


def configurationPATH(cf):
    '''
    :param cf: config file
    :return: Print some paths
    '''

    print("\n###########################")
    print(' > Conf File Path = "%s"' % (cf.config_path))
    print(' > Save Path = "%s"' % (cf.savepath))
    print(' > Dataset PATH = "%s"' % (os.path.join(cf.dataroot_dir)))
    print("###########################\n")


def show_DG(DG, show_set='train'):
    # Let's instantiate this class the iterate through the data samples.
    train_set = DG.dataloader[show_set]

    for i_batch, sample_batched in enumerate(train_set):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())

        # observe 4th batch and stop.
        if i_batch == 0:
            plt.figure()
            show_landmarks_batch(sample_batched, DG)
            #plt.axis('off')
            plt.ioff()
            plt.show()
            break


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
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
    plt.title('Synthetic images')
    plt.subplot(2, 1, 2)
    plt.imshow(grid_labels.numpy().transpose(1, 2, 0)[:,:,0], cmap='jet')
    plt.title('Masks')


def show_DG_car_trajectory(DG, show_set='train'):
    # Let's instantiate this class the iterate through the data samples.
    train_set = DG.dataloader[show_set]

    for i_batch, sample_batched in enumerate(train_set):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['classes'].size(),
              sample_batched['instances'].size())

        # observe 4th batch and stop.
        if i_batch == 2:
            plt.figure()
            show_landmarks_batch_car_trajectory(sample_batched, DG)
            im_name_tmp = "-".join(sample_batched['img_name'][0].split("/")[-5:])
            save_im_name = os.path.join('/home/stevenwudi/PycharmProjects/autonomous_driving/Experiments/car_trajectory_prediction/Figures', im_name_tmp)
            plt.savefig(save_im_name, bbox_inches='tight', dpi=1000)
            #plt.axis('off')
            plt.ioff()
            #plt.ion()
            plt.show()
            #plt.waitforbuttonpress()
            break


# Helper function to show a batch
def show_landmarks_batch_car_trajectory(sample_batched, DG):
    """Show image with landmarks for a batch of samples."""
    images_batch, classes_batch, instances_batch = \
            sample_batched['image'], sample_batched['classes'], sample_batched['instances']

    grid_images = utils.make_grid(images_batch, nrow=5, padding=10)
    # because the utils.make_grid requires tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
    # also the grid image is of dtype DoubleTensor
    classes_batch = classes_batch.unsqueeze(1).type(torch.DoubleTensor)
    grid_classes = utils.make_grid(classes_batch, nrow=5, padding=10, normalize=True)

    instances_batch = instances_batch.unsqueeze(1).type(torch.DoubleTensor)
    grid_instances = utils.make_grid(instances_batch, nrow=5, padding=10, normalize=True)
    # we need to expand the labels range
    plt.subplot(3, 1, 1)
    plt.imshow(grid_images.numpy().transpose(1, 2, 0))
    plt.title('Synthetic images')
    plt.subplot(3, 1, 2)
    plt.imshow(grid_classes.numpy().transpose(1, 2, 0)[:,:,0], cmap='jet')
    #plt.title('classes')
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(grid_instances.numpy().transpose(1, 2, 0)[:,:,0], cmap='jet')
    plt.title('instances')
    plt.axis('off')