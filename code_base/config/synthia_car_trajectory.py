# Dataset
problem_type                 = 'car_trajectory_prediction'  # ['classification' | 'detection' | 'segmentation']
#dataset_name                 = 'synthia_rand_cityscapes'        # Dataset
sequence_name                = 'SYNTHIA-SEQS-06'
local_path                   = '/home/stevenwudi/PycharmProjects/autonomous_driving'
shared_path                  = '/home/public'
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch
class_mode                   = problem_type

# Model
model_name                   = 'fcn8'          # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']
debug                        = False
resize_train                 = (760, 1280)      # Resize the image during training (Height, Width) or None
#random_size_crop             = (350*2, 460*2)      # Random size crop of the image during training
batch_size_train             = 1              # Batch size during training
batch_size_valid             = 8              # Batch size during validation
batch_size_test              = 8              # Batch size during testing
dataloader_num_workers_train = 1        # Number of dataload works during training
dataloader_num_workers_valid = 1        # Number of dataload works during valid
dataloader_num_workers_test  = 1        # Number of dataload works during test
# Data shuffle
shuffle_train                = False            # No shuffling because the time sequence matters
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data


# Parameters
train_model                  = True            # Train the model
test_model                   = True           # Test the model
pred_model                   = False           # Predict using the model


# Training parameters
optimizer                    = 'adam'       # Optimizer
learning_rate                = 0.0001          # Training learning rate
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty
n_epochs                     = 1            # Number of epochs during training

#############################
dataroot_dir                        = '/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets'
data_type                           = 'RGB'
data_stereo                         = 'Stereo_Left'
data_camera                         = 'Omni_F'
data_label                          = 'LABELS'  # we have 'COLORS' and 'LABELS'
norm_fit_dataset                    = False   # If True it recompute std and mean from images. Either it uses the std and mean set at the dataset config file
norm_featurewise_center             = False   # Substract mean - dataset
norm_featurewise_std_normalization  = False   # Divide std - dataset
color_mode                          = 'rgb'
n_channels                          = 3
rgb_mean                            = [0.39450742,  0.37999875,  0.35578521] # Wudi pre-computed mean using first 1000 images
rgb_std                             = [0.20455311,  0.20075491,  0.19981377] # Wudi pre-computed mean using first 1000 images
#classes                             = ['void', 'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist']
classes                             = {'void': 0, 'sky': 1, 'building': 2, 'road': 3,
                                       'sidewalk': 4, 'fence': 5, 'vegetation': 6,
                                       'pole': 7, 'car': 8, 'sign': 9, 'pedestrian': 10,
                                       'cyclist': 11}
n_classes                           = len(classes)
void_class                          = [len(classes) + 1]
create_split                        = False
cb_weights_method                   = 'rare_freq_cost'   # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']
cb_weights                          = [5.31950559,   1.65697855,   0.23748228,   0.29841721,
         0.63769955,   9.23991394,   1.66974087,   6.60188582,
         0.92809024,  19.85701845,   2.60712632,  14.72396384]

threshold_car_POR_start                 = 2e-3  # we only count if the sum of the pixels are larger than 1e-3
threshold_car_POR_end                   = 1e-3  # we only count if the sum of the pixels are larger than 1e-3