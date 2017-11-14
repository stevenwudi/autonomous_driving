# Dataset
problem_type                 = 'segmentation'  # ['classification' | 'detection' | 'segmentation']



dataset_name                 = 'SYNTHIA_RAND_CVPR16'

dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch
class_mode                   = problem_type
local_path                   = '/home/stevenwudi/PycharmProjects/autonomous_driving'
shared_path                  = '/media/samsumg_1tb/synthia'

dataroot_dir                 = '/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets'


# Model

model_name                   = 'segnet_basic'  # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']

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
random_size_crop             = (512, 512)      # Random size crop of the image during training
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

learning_rate                = 0.0001          # Training learning rate
weight_decay                 = 1e-4              # Weight decay or L2 parameter norm penalty   1e-4
n_epochs                     = 10            # Number of epochs during training
momentum                     = 0.9  #0.9

# Callback save results
save_results_enabled         = False           # Enable the Callback
save_results_nsamples        = 5               # Number of samples to save
save_results_batch_size      = 5               # Size of the batch


# Callback early stoping
earlyStopping_enabled        = True            # Enable the Callback
earlyStopping_monitor        = 'val_jaccard'   # Metric to monitor
earlyStopping_mode           = 'max'           # Mode ['max' | 'min']
earlyStopping_patience       = 100             # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping

# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'val_jaccard'   # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = True            # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 0               # Verbosity of the checkpoint

# Callback plot
plotHist_enabled             = True            # Enable the Callback
plotHist_verbose             = 0               # Verbosity of the callback

# Callback LR decay scheduler
lrDecayScheduler_enabled     = False           # Enable the Callback
lrDecayScheduler_epochs      = [5, 10, 20]     # List of epochs were decay is applied or None for all epochs
lrDecayScheduler_rate        = 2               # Decay rate (new_lr = lr / decay_rate). Usually between 2 and 10.

# Callback learning rate scheduler
LRScheduler_enabled          = True             # Enable the Callback
LRScheduler_batch_epoch      = 'batch'          # Schedule the LR each 'batch' or 'epoch'
LRScheduler_type             = 'poly'         # Type of scheduler ['linear' | 'step' | 'square' | 'sqrt' | 'poly']
LRScheduler_M                = 75000            # Number of iterations/epochs expected until convergence
LRScheduler_decay            = 0.1              # Decay for 'step' method
LRScheduler_S                = 10000            # Step for the 'step' method
LRScheduler_power            = 0.9              # Power for te poly method

# Callback TensorBoard
TensorBoard_enabled          = True             # Enable the Callback
TensorBoard_histogram_freq   = 1                # Frequency (in epochs) at which to compute activation histograms for the layers of the model. If set to 0, histograms won't be computed.
TensorBoard_write_graph      = True             # Whether to visualize the graph in Tensorboard. The log file can become quite large when write_graph is set to True.
TensorBoard_write_images     = False            # Whether to write model weights to visualize as image in Tensorboard.
TensorBoard_logs_folder      = None             #

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

# mean                                = [0.290101, 0.328081, 0.286964]
# std                                 = [0.182954, 0.186566, 0.184475]


load_trained_model                  = True
# train_model_path                    = '/home/ty/code/autonomous_driving/Experiments/segmentation/synthia_rand_cityscapes___Mon, 06 Nov 2017-11-06 22:17:51/25_net.pth'
# train_model_path                    = '/home/ty/code/autonomous_driving/Experiments/CityScape_semantic_segmentation/drn_d_38_cityscapes.pth'
train_model_path                    = '/home/ty/code/autonomous_driving/Experiments/segmentation/synthia_rand_cityscapes___Tue, 07 Nov 2017-11-07 13:02:10/5_net.pth'
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

ignore_label                        = 0

create_split                        = False

cb_weights_method                   = 'rare_freq_cost'   # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']
cb_weights                          = [5.31950559,   1.65697855,   0.23748228,   0.29841721,
         0.63769955,   9.23991394,   1.66974087,   6.60188582,
         0.92809024,  19.85701845,   2.60712632,  14.72396384]

error_images                        = ['ap_000_02-11-2015_18-02-19_000062_3_Rand_2.png', 'ap_000_02-11-2015_18-02-19_000129_2_Rand_16.png',
                                        'ap_000_01-11-2015_19-20-57_000008_1_Rand_0.png']
test_epoch                          = 5

full_to_colour = {0: (0, 0, 0), 1: (128, 128, 128), 2: (128, 0, 0), 3: (128, 64, 128),
                  4: (0, 0, 192), 5: (64, 64, 128), 6: (128, 128, 0), 7: (192, 192, 128),
                  8: (64, 0, 128), 9: (192, 192, 128), 10: (64, 64, 0), 11: (0, 128, 192)}
# full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70),
#                   12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30),
#                   20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180),
#                   24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60, 100),
#                   31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}

ignore_index                        = 0
test_epoch                          = 1
batch_size                          = 1
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
train_model_path             = '/home/stevenwudi/PycharmProjects/autonomous_driving/Experiments/segmentation/SYNTHIA-SEQS-01___Sun, 12 Nov 2017-11-12 21:57:56_drn_d_38/epoch_21_mIOU:.0.339176_net.pth'
data_type                           = 'RGB'
data_stereo                         = 'Stereo_Left'
data_camera                         = 'Omni_F'
data_label                          = 'LABELS'  # we have 'COLORS' and 'LABELS'
start_tracking_idx           = 1
resize_train                 = (720, 960)    # Resize the image during training (Height, Width) or None

dataset_name                 = 'SYNTHIA_RAND_CVPR16'
sequence_name                = 'SYNTHIA-SEQS-01'
#sequence_name = dataset_name
