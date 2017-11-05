# Dataset
problem_type                 = 'segmentation'  # ['classification' | 'detection' | 'segmentation']
dataset_name                 = 'cityscapes'        # Dataset name
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch

# Model


model_name                   = 'drn_d_38'  # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']

                                               # ty add: drn_c_26 drn_c_42, drn_d_22 .....
freeze_layers_from           = None            # Freeze layers from 0 to this layer during training (Useful for finetunning) [None | 'base_model' | Layer_id]
show_model                   = True            # Show the architecture layers
load_imageNet                = False           # Load Imagenet weights and normalize following imagenet procedure
load_pretrained              = False            # Load a pretrained model for doing finetuning

# Parameters
train_model                  = True            # Train the model
test_model                   = True            # Test the model
pred_model                   = False           # Predict using the model

# Debug
debug                        = False            # Use only few images for debuging

# Batch sizes

# Data shuffle
shuffle_train                = True            # Whether to shuffle the training data
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data
seed_train                   = 1924            # Random seed for the training shuffle
seed_valid                   = 1924            # Random seed for the validation shuffle
seed_test                    = 1924            # Random seed for the testing shuffle

# Training parameters
optimizer                    = 'sgd'       # Optimizer

learning_rate                = 0.001  #0.001, 0.0001          # Training learning rate
momentum                     = 0.99  #0.9
weight_decay                 = 1e-4              # Weight decay or L2 parameter norm penalty
n_epochs                     = 250            # Number of epochs during training



# Data
dataroot_dir                        = '/home/public/CITYSCAPE/'
crop_size                           = 896
workers                             = 8  #'Data loader workers'

num_classes                         = 12
class_mode                          = 'segmentation'
batch_size                          = 6

load_trained_model                  = True
train_model_path                    = '/home/ty/code/autonomous_driving/Experiments/CityScape_semantic_segmentation/drn_d_38_cityscapes.pth'

mean                                = [0.290101, 0.328081, 0.286964]
std                                 = [0.182954, 0.186566, 0.184475]
pretrained_drn_c_26                 = True
test_epoch                          = 2
# Labels: -1 license plate, 0 unlabeled, 1 ego vehicle, 2 rectification border, 3 out of roi,
# 4 static, 5 dynamic, 6 ground, 7 road, 8 sidewalk, 9 parking, 10 rail track, 11 building,
# 12 wall, 13 fence, 14 guard rail, 15 bridge, 16 tunnel, 17 pole, 18 polegroup, 19 traffic light,
# 20 traffic sign, 21 vegetation, 22 terrain, 23 sky, 24 person, 25 rider, 26 car, 27 truck, 28 bus,
# 29 caravan, 30 trailer, 31 train, 32 motorcycle, 33 bicycle

full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2,
                 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24,
                 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70),
                  12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30),
                  20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180),
                  24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60, 100),
                  31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}
