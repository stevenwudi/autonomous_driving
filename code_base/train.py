#!/usr/bin/env python
import argparse
import os
import sys
from getpass import getuser
import matplotlib
import time
from datetime import datetime
#matplotlib.use('Agg')  # Faster plot

# Import tools
from code_base.config.configuration import Configuration
from code_base.tools.logger import Logger
from code_base.tools.dataset_generators import Dataset_Generators
from code_base.tools.optimizer_factory import Optimizer_Factory
from code_base.callbacks.callbacks_factory import Callbacks_Factory
from code_base.models.model_factory import Model_Factory


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

    print ("\n###########################")
    print (' > Conf File Path = "%s"' % (cf.config_path))
    print (' > Save Path = "%s"' % (cf.savepath))
    print (' > Dataset PATH = "%s"' % (os.path.join(dataset_path, cf.problem_type, cf.dataset_name)))
    print ("###########################\n")


# Train the network
def process(cf):
    # Enable log file
    sys.stdout = Logger(cf.log_file)
    print (' ---> Init experiment: ' + cf.exp_name + ' <---')

    # Create the data generators
    train_gen, valid_gen, test_gen = Dataset_Generators().make(cf)

    # Create the optimizer
    print ('\n > Creating optimizer...')
    optimizer = Optimizer_Factory().make(cf)

    # Build model
    print ('\n > Building model...')
    model = Model_Factory().make(cf, optimizer)

    # Create the callbacks
    print ('\n > Creating callbacks...')
    cb = Callbacks_Factory().make(cf, valid_gen)

    if cf.train_model:
        # Train the model
        model.train(train_gen, valid_gen, cb)

    if cf.test_model:
        # Compute validation metrics
        model.test(valid_gen)
        # Compute test metrics
        model.test(test_gen)

    if cf.pred_model:
        # Compute validation metrics
        model.predict(valid_gen, tag='pred')
        # Compute test metrics
        model.predict(test_gen, tag='pred')

    # Finish
    print (' ---> Finish experiment: ' + cf.exp_name + ' <---')


# Sets the backend and GPU device.
class Environment():
    def __init__(self, backend='tensorflow'):
        #backend = 'tensorflow' # 'theano' or 'tensorflow'
        os.environ['KERAS_BACKEND'] = backend
        os.environ["CUDA_VISIBLE_DEVICES"]="0" # "" to run in CPU, extra slow! just for debuging
        if backend == 'theano':
            # os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'
            """ fast_compile que lo que hace es desactivar las optimizaciones => mas lento """
            os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.95'
            print('Backend is Theano now')
        else:
            print('Backend is Tensorflow now')


# Main function
def main():
    # Define environment variables
    # Environment()

    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='Configuration file')
    parser.add_argument('-e', '--exp_name', type=str,
                        default=None, help='Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default='/data', help='Path to shared data folder')
    parser.add_argument('-l', '--local_path', type=str,
                        default='/datatmp', help='Path to local data folder')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration'\
                                              'path using -c config/pathname'\
                                              ' in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the '\
                                           'experiment using -e name in the '\
                                           'command line'

    # Start Time
    print ('\n > Start Time:')
    print ('   '+ datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()

    # Define the user paths
    shared_path = arguments.shared_path
    local_path = arguments.local_path
    dataset_path = os.path.join(local_path, 'Datasets')
    shared_dataset_path = os.path.join(shared_path)
    experiments_path = os.path.join(local_path, 'Experiments')
    shared_experiments_path = os.path.join(shared_path, 'Experiments')
    usr_path = os.path.join('/home/', getuser())

    # Load configuration files
    configuration = Configuration(arguments.config_path, arguments.exp_name,
                                  dataset_path, shared_dataset_path,
                                  experiments_path, shared_experiments_path)

    cf = configuration.load()

    configurationPATH(cf, dataset_path)

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # Copy result to shared directory
    # configuration.copy_to_shared()

    # End Time
    end_time = time.time()
    print ('\n > End Time:')
    print ('   '+ datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print ('\n   ET: '+ HMS(end_time - start_time)) # -> H:M:S


# Entry point of the script
if __name__ == "__main__":
    main()
