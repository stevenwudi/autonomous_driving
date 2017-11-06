# Dataset
problem_type                 = 'car_trajectory_prediction'  # ['classification' | 'detection' | 'segmentation']
#dataset_name                 = 'synthia_rand_cityscapes'        # Dataset

local_path                   = '/home/wzn/PycharmProjects/autonomous_driving'
shared_path                  = '/home/public/synthia'
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch
class_mode                   = problem_type

sequence_name                = 'SYNTHIA-SEQS-01'
collect_data                 = False
get_ground_truth_sequence_car_trajectory = False  # flag to get get_ground_truth_sequence_car_trajectory
formatting_ground_truth_sequence_car_trajectory = True
draw_seq                                = 'SYNTHIA-SEQS-06-NIGHT'   # which sequence to draw, need to set the above two flags to False


# Model
model_name                   = 'LSTM_To_FC'       # Model to use ['LSTM_ManyToMany', 'LSTM_To_FC']
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

#############################
dataroot_dir                        = '/home/wzn/PycharmProjects/autonomous_driving/Datasets'
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

threshold_car_POR_start                 = 2e-3  # threshold for car start tracking using pixel occupant rate
threshold_car_POR_end                   = 1e-3  # threshold for car start tracking using pixel occupant rate
threshold_car_depth                     = False  # flag for deciding depth as the start of car tracking
threshold_car_depth_start               = 2000  # threshold for car start tracking using depth
threshold_car_depth_end                 = 3000  # threshold for car end tracking using depth

lstm_training_stepsize                  = [1, 23]  # step size for collecting training data
lstm_input_frame                        = 15
lstm_predict_frame                      = 8

# Parameters
train_model                  = True            # Train the model
valid_model                  = True           # Test the model
test_model                   = True           # Predict using the model

# camera intrinsics
focal_length                 = 532.740352  # camera focal lense

# Training parameters
test_epoch                   = 1
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty
n_epochs                     = 3            # Number of epochs during training
cuda                         = True
loss                         = 'SmoothL1Loss'       # 'MSE', 'SmoothL1Loss'
optimizer                    = 'LBFGS'      # LBFGS','adam'
learning_rate                = 0.1          # Training learning rate
momentum                     = 0.9
load_trained_model           = True
train_model_path             = '/home/stevenwudi/PycharmProjects/autonomous_driving/Experiments/car_trajectory_prediction/SYNTHIA-SEQS-01___Mon, 06 Nov 2017-11-06 16:08:14/Epoch:100_net_aveErrCoverage:0.8343_aveErrCenter:17.47___.pth'
#### LSTM training variables #################
# LSTM_ManyToMany
lstm_inputsize               = 6   # LSTM input: [x,y,w,h, d_min, d_max]
lstm_hiddensize              = 50
lstm_hiddensize              = 50
lstm_numlayers               = 2
lstm_outputsize              = 6
# LSTM_To_FC
lstm_output_dim              = 6   # currently is [x,y,w,h,d_min, d_max] as lstm_inputsize
