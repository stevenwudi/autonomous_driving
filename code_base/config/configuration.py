import imp
import time
import os
from distutils.dir_util import copy_tree
import shutil


class Configuration():
    def __init__(self, config_path):

        self.config_path = config_path

    def load(self):
        # Load configuration file...
        print(self.config_path)
        cf = imp.load_source('config', self.config_path)
        dataset_path = os.path.join(cf.local_path, 'Datasets')
        experiments_path = os.path.join(cf.local_path, 'Experiments')
        shared_experiments_path = os.path.join(cf.shared_path, 'Experiments')

        # Save extra parameter
        cf.config_path = self.config_path
        cf.exp_name = cf.problem_type

        # If in Debug mode use few images
        if cf.debug and cf.debug_images_train > 0:
            cf.dataset.n_images_train = cf.debug_images_train
        if cf.debug and cf.debug_images_valid > 0:
            cf.dataset.n_images_valid = cf.debug_images_valid
        if cf.debug and cf.debug_images_test > 0:
            cf.dataset.n_images_test = cf.debug_images_test
        if cf.debug and cf.debug_n_epochs > 0:
            cf.n_epochs = cf.debug_n_epochs

        # Plot metrics
        if cf.class_mode == 'segmentation':
            cf.train_metrics = ['loss', 'acc', 'jaccard']
            cf.valid_metrics = ['val_loss', 'val_acc', 'val_jaccard']
            cf.best_metric = 'val_jaccard'
            cf.best_type = 'max'
        elif cf.class_mode == 'car_trajectory_prediction':
            cf.train_metrics = ['loss', 'acc', 'jaccard']
            cf.valid_metrics = ['val_loss', 'val_acc', 'val_jaccard']
            cf.best_metric = 'val_jaccard'
            cf.best_type = 'max'
        elif cf.class_mode == 'detection':
            # TODO detection : different nets may have other metrics
            cf.train_metrics = ['loss', 'avg_recall', 'avg_iou']
            cf.valid_metrics = ['val_loss', 'val_avg_recall', 'val_avg_iou']
            cf.best_metric = 'val_avg_recall'
            cf.best_type = 'max'
        else:
            cf.train_metrics = ['loss', 'acc']
            cf.valid_metrics = ['val_loss', 'val_acc']
            cf.best_metric = 'val_acc'
            cf.best_type = 'max'

        self.configuration = cf
        if cf.sequence_name:
            if len(cf.sequence_name) == 15:
                # it means we take the SYNTHIA-SEQS-0* as the sequence_name
                cf.dataset_path = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if x[:15]==cf.sequence_name]
                cf.savepath = os.path.join(experiments_path, cf.exp_name, cf.sequence_name)
                if not os.path.exists(experiments_path):
                    os.mkdir(experiments_path)
                if not os.path.exists(os.path.join(experiments_path, cf.exp_name)):
                    os.mkdir(os.path.join(experiments_path, cf.exp_name))
                if not os.path.exists(cf.savepath):
                    os.mkdir(cf.savepath)

            else:
                cf.dataset_path = os.path.join(dataset_path, cf.problem_type, cf.sequence_name)
                # Create output folders
                cf.savepath = os.path.join(experiments_path, cf.exp_name, cf.dataset_name)

                cf.final_savepath = os.path.join(shared_experiments_path, cf.dataset_name,
                                                 cf.exp_name)
                # cf.log_file = os.path.join(cf.savepath, "logfile.log")
                if not os.path.exists(experiments_path):
                    os.mkdir(experiments_path)
                if not os.path.exists(os.path.join(experiments_path, cf.exp_name)):
                    os.mkdir(os.path.join(experiments_path, cf.exp_name))
                # if not os.path.exists(cf.savepath):
                #     os.mkdir(cf.savepath)
        else:
            cf.dataset_path = os.path.join(dataset_path, cf.problem_type, cf.dataset_name)
        return cf

    # Load the configuration file of the dataset
    def load_config_dataset(self, savepath, dataset_name, dataset_path, shared_dataset_path, problem_type, name='config'):
        # Copy the dataset from the shared to the local path if not existing
        #shared_dataset_path = os.path.join(shared_dataset_path, problem_type, dataset_name)
        dataset_path = os.path.join(dataset_path, problem_type, dataset_name)
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        # Load dataset config file
        #dataset_config_path = os.path.join(savepath, 'config.py')
        dataset_config_path = self.config_path

        print('dataset_config_path', dataset_config_path)
        dataset_conf = imp.load_source(name, dataset_config_path)
        dataset_conf.config_path = dataset_config_path

        # Compose dataset paths
        dataset_conf.path = dataset_path
        if dataset_conf.class_mode == 'segmentation':
            dataset_conf.path_train_img = os.path.join(dataset_conf.path, 'train', 'images')
            dataset_conf.path_train_mask = os.path.join(dataset_conf.path, 'train', 'masks')
            dataset_conf.path_valid_img = os.path.join(dataset_conf.path, 'valid', 'images')
            dataset_conf.path_valid_mask = os.path.join(dataset_conf.path, 'valid', 'masks')
            dataset_conf.path_test_img = os.path.join(dataset_conf.path, 'test', 'images')
            dataset_conf.path_test_mask = os.path.join(dataset_conf.path, 'test', 'masks')
        else:
            dataset_conf.path_train_img = os.path.join(dataset_conf.path, 'train')
            dataset_conf.path_train_mask = None
            dataset_conf.path_valid_img = os.path.join(dataset_conf.path, 'valid')
            dataset_conf.path_valid_mask = None
            dataset_conf.path_test_img = os.path.join(dataset_conf.path, 'test')
            dataset_conf.path_test_mask = None

        return dataset_conf

    # Copy result to shared directory
    def copy_to_shared(self):
        if self.configuration.savepath != self.configuration.final_savepath:
            print('\n > Copying model and other training files to {}'.format(self.configuration.final_savepath))
            start = time.time()
            copy_tree(self.configuration.savepath, self.configuration.final_savepath)
            open(os.path.join(self.configuration.final_savepath, 'lock'), 'w').close()
            print ('   Copy time: ' + str(time.time()-start))
