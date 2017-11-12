# Dataset
problem_type                 = 'segmentation'  # ['classification' | 'detection' | 'segmentation']

dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch
class_mode                   = problem_type
local_path                   = '/home/stevenwudi/PycharmProjects/autonomous_driving'
shared_path                  = '/media/samsumg_1tb/synthia'

dataroot_dir                 = '/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets'


# Model
model_name                   = 'drn_d_38'  # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']
freeze_layers_from           = None            # Freeze layers from 0 to this layer during training (Useful for finetunning) [None | 'base_model' | Layer_id]
show_model                   = False           # Show the architecture layers
load_imageNet                = False           # Load Imagenet weights and normalize following imagenet procedure

# Parameters
train_model                  = True            # Train the model
test_model                   = True           # Test the model
pred_model                   = False           # Predict using the model

# Debug
debug                        = False           # Use only few images for debuging
debug_images_train           = 50              # N images for training in debug mode (-1 means all)
debug_images_valid           = 30              # N images for validation in debug mode (-1 means all)
debug_images_test            = 30              # N images for testing in debug mode (-1 means all)
debug_n_epochs               = 3               # N of training epochs in debug mode

# Batch sizes

workers                      = 4
batch_size_train             = 8              # Batch size during training
batch_size_valid             = 8              # Batch size during validation
batch_size_test              = 8              # Batch size during testing
dataloader_num_workers_train = batch_size_train# Number of dataload works during training
dataloader_num_workers_valid = batch_size_valid# Number of dataload works during valid
dataloader_num_workers_test  = batch_size_test # Number of dataload works during test

crop_size_train              = (224, 224)      # Crop size during training (Height, Width) or None
crop_size_valid              = None            # Crop size during validation
crop_size_test               = None            # Crop size during testing
resize_train                 = (270, 480)      # Resize the image during training (Height, Width) or None
resize_valid                 = (270, 480)      # Resize the image during validation
resize_test                  = (270, 480)      # Resize the image during testing
random_size_crop             = (250, 450)      # Random size crop of the image during training
#random_size_crop             = None     # Random size crop of the image during training


# Data shuffle
shuffle_train                = True            # Whether to shuffle the training data
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data
seed_train                   = 1924            # Random seed for the training shuffle
seed_valid                   = 1924            # Random seed for the validation shuffle
seed_test                    = 1924            # Random seed for the testing shuffle

# Training parameters
optimizer                    = 'sgd'       # Optimizer
learning_rate                = 0.001          # Training learning rate
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty
n_epochs                     = 100            # Number of epochs during training
momentum                     = 0.9

# Data augmentation for training and normalization
norm_imageNet_preprocess           = False  # Normalize following imagenet procedure
norm_rescale                       = 1/255. # Scalar to divide and set range 0-1

norm_samplewise_center             = False  # Substract mean - sample
norm_samplewise_std_normalization  = False  # Divide std - sample
norm_gcn                           = False  # Global contrast normalization
norm_zca_whitening                 = False  # Apply ZCA whitening

# Data augmentation for training
da_rotation_range                  = 0      # Rnd rotation degrees 0-180
da_width_shift_range               = 0.0    # Rnd horizontal shift
da_height_shift_range              = 0.0    # Rnd vertical shift
da_shear_range                     = 0.0    # Shear in radians
da_zoom_range                      = 0.0    # Zoom
da_channel_shift_range             = 0.     # Channecf.l shifts
da_fill_mode                       = 'constant'  # Fill mode
da_cval                            = 0.     # Void image value
da_horizontal_flip                 = False  # Rnd horizontal flip
da_vertical_flip                   = False  # Rnd vertical flip
da_spline_warp                     = False  # Enable elastic deformation
da_warp_sigma                      = 10     # Elastic deformation sigma
da_warp_grid_size                  = 3      # Elastic deformation gridSize
da_save_to_dir                     = False  # Save the images for debuging

#############################
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
classes = {'Void': 0, 'Sky': 1, 'Building': 2, 'Road': 3, 'Sidewalk': 4, 'Fence': 5, 'Vegetation': 6, 'Pole': 7,
           'Car': 8, 'Traffic_Sign': 9, 'Pedestrian': 10, 'Bicycle': 11, 'Lanemarking': 12,
           'Reserved': 13, 'Reserved': 14, 'Traffic Light': 15}
num_classes                           = len(classes) +1   # we don't want void class
#num_classes                           = len(classes)   # we don't want void class
void_class                          = [len(classes) + 1]
create_split                        = False
ignore_index                        = 0
test_epoch                          = 1
batch_size                          = 8
crop_size                           = 720
train_ratio                         = 0.9
cb_weights_method                   = None  #'rare_freq_cost'# Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']
cb_weights                          = [1.65697855,   0.23748228,   0.29841721, 0.63769955,   9.23991394,   1.66974087,
                                       6.60188582, 0.92809024,  19.85701845,   2.60712632,  14.72396384]


video_sequence_train         = True
train_ratio                  = 0.8
valid_ratio                  = 0.1
video_sequence_prediction    = False

load_trained_model           = False           # Load a pretrained model for doing finetuning
train_model_path             = '/home/stevenwudi/PycharmProjects/autonomous_driving/Experiments/segmentation/SYNTHIA_RAND_CVPR16___Wed, 08 Nov 2017-11-08 11:21:17_drn_d_38/epoch_44_mIOU:.0.603778_net.pth'
data_type                           = 'RGB'
data_stereo                         = 'Stereo_Left'
data_camera                         = 'Omni_F'
data_label                          = 'LABELS'  # we have 'COLORS' and 'LABELS'
start_tracking_idx           = 1
resize_train                 = (720, 960)    # Resize the image during training (Height, Width) or None

dataset_name                 = 'SYNTHIA_RAND_CVPR16'
sequence_name                = 'SYNTHIA-SEQS-01'
#sequence_name = dataset_name