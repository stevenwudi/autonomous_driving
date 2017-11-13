############################# Overall logic here
problem_type                 = 'car_trajectory_prediction'  # ['car_tracking', 'car_detection', 'car_trajectory_prediction']
sequence_name                = 'SYNTHIA-SEQS-01'

car_detection                = False
car_tracking                 = True
tracker                      = 'dlib_dsst'  #['dlib_dsst', 'ECO_HC', 'KCF']
draw_flag                    = True
car_detection_method         = 'ssd512'
get_sequence_car_detection   = True
get_sequence_car_tracking    = True
depth_threshold_method       = 'yen'   #['yen', 'ostu', 'li'
iou_threshold                = 0.3
start_tracking_idx           = 850
threshold_car_POR_start      = 2e-3  # threshold for car start tracking using pixel occupant rate
threshold_car_POR_end        = 1e-3  # threshold for car start tracking using pixel occupant rate
minimum_detection_length     = 3

####  SSD parameters (currently, we only suppose keras with Tensorflow backend, TODO: use Pytorch!
ssd_prior_boxes              = '/home/stevenwudi/PycharmProjects/autonomous_driving/code_base/models/prior_boxes_ssd512.pkl'
ssd_number_classes           = 2
ssd_model_checkpoint         = '/home/public/synthia/ssd_car_fine_tune/weights_512.54-0.19.hdf5'
ssd_input_shape              = (512, 512, 3)
ssd_conf                     = 0.5



local_path                   = '/home/wzn/PycharmProjects/autonomous_driving'
# Dataset
shared_path                  = '/home/public/synthia'
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch
class_mode                   = problem_type
collect_data                 = False
get_ground_truth_sequence_car_trajectory = True  # flag to get get_ground_truth_sequence_car_trajectory
formatting_ground_truth_sequence_car_trajectory = True
draw_seq                                = 'SYNTHIA-SEQS-06-NIGHT'   # which sequence to draw, need to set the above two flags to False

# Model
model_name                   = 'CNN_LSTM_To_FC'       # Model to use ['LSTM_ManyToMany', 'LSTM_To_FC', 'CNN_LSTM_To_FC']
debug                        = False
im_size                      = (760, 1280)
resize_train                 = (760, 1280)      # Resize the image during training (Height, Width) or None
#random_size_crop             = (350*2, 460*2)      # Random size crop of the image during training
#------ DataLoader ------#
workers                      = 32              #  how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process
batch_size_train             = 64            # Batch size during training
batch_size_valid             = 20              # Batch size during validation
batch_size_test              = 20              # Batch size during testing
dataloader_num_workers_train = 1        # Number of dataload works during training
dataloader_num_workers_valid = 1        # Number of dataload works during valid
dataloader_num_workers_test  = 1        # Number of dataload works during test
dataloader_load_prepare_data = False     # whether or not load prepare_data for DataLoader
dataloader_load_prepare_data_path = '/media/samsumg_1tb/synthia/SYNTHIA-SEQS-01/prepared_data.npy'
dataloader_save_prepare_data = True     # whether or not save generated prepare_data for DataLoader
dataloader_save_prepare_data_path = '/media/samsumg_1tb/synthia/SYNTHIA-SEQS-01/prepared_data.npy'
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
lr_decay_epoch               = 10          # every lr_decay_epoch, decay learning_rate by 10
figure_epoch                 = 5           # every fugure_epoch, figure out the losses
test_epoch                   = 1
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty
n_epochs                     = 200            # Number of epochs during training
cuda                         = True
loss                         = 'SmoothL1Loss'       # 'MSE', 'SmoothL1Loss'
optimizer                    = 'sgd'      # LBFGS','adam','rmsprop','sgd'
learning_rate                = 0.001          # Training learning rate
momentum                     = 0.9
load_trained_model           = True
train_model_path             = '/home/wzn/PycharmProjects/autonomous_driving/Experiments/car_trajectory_prediction/SYNTHIA-SEQS-01_Mon, 13 Nov 2017-11-13 11:27:35_CNN_LSTM_To_FC/Epoch: 5_net_Coverage:0.0859_Center:231.26_CoverageR:0.0423_CenterR:1.22.PTH'
#### LSTM training variables #################
# LSTM_ManyToMany
lstm_input_dims               = [6, 150, 150]    # [layer1_input_dim, layer2_input_dim,...]  layer1_input_dim:[x,y,w,h, d_min, d_max]
lstm_hidden_sizes             = [150, 150, 150]    # [layer1_hidden_size, layer2_hidden_size,...]
outlayer_input_dim            = 150          # outlayer's input dim.Generally, identify to hidden_sizes[-1]
outlayer_output_dim           = 6            # outlayer output: [x,y,w,h, d_min, d_max]
# LSTM_To_FC
lstmToFc_input_dims           = [6, 100, 300]              # [layer1_input_dim, layer2_input_dim,...]  layer1_input_dim:[x,y,w,h, d_min, d_max]
lstmToFc_hidden_sizes         = [100, 300, 300]            # [layer1_hidden_size, layer2_hidden_size,...]
lstmToFc_future               = lstm_predict_frame        # the number of predicting frames
lstmToFc_output_dim           = 6               # outlayer output: [x,y,w,h, d_min, d_max]

# CNN_LSTM_To_FC
cnn_class_num                 = 15
def cnnDict(in_channels, out_channels, kernel_size, stride, padding):
    return {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
cnnLstmToFc_conv_paras        = [cnnDict(cnn_class_num,15,3,1,1), cnnDict(15,10,3,1,1),cnnDict(10,5,3,1,1),cnnDict(5,5,3,1,1)]              # a list composed of dicts representing parameters of each conv, {'in_channels': ,
                                                                                      # 'out_channels': ,
                                                                                      # 'kernel_size': ,
                                                                                      # 'stride': ,
                                                                                      # 'padding': }
cnnLstmToFc_input_dims        = [6, 350, 350]              # a list involving each lstm_layer's input_dim
cnnLstmToFc_hidden_sizes      = [100, 350, 350]              # a list involving each lstm_layer's hidden_size
cnnLstmToFc_future            = lstm_predict_frame # the number of predicting frames
cnnLstmToFc_output_dim        = 6               # outlayer output: [x,y,w,h, d_min, d_max]
