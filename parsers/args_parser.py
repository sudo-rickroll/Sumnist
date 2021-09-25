import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Add MNIST with Random Digit')
    parser.add_argument('--mode', default='train', help='Train Model')
    parser.add_argument('--config_path', default='./config/config_file.cfg', help='Path of config file')
    parser.add_argument('--checkpoint_save', default='', help='Save checkpoint of model')
    parser.add_argument('--checkpoint_load', default='', help='Load checkpoint of model')
    args = parser.parse_args()
    return args