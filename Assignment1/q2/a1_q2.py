"""
:author: Cameron Sims
:date: 09/09/2025
:description: This is the VGG16 implementation
"""
import torch 
import torch.nn as nn

import model

class VGG16(nn.Module): 
    """
    :description: VGG16 class, used to implement CNN architecture.
    """

    def __init__(self, input_amount: int, output_amount: int):


        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def main():
    """
    :description: This is where we start from.
    """
    from os import getcwd as os_getcwd
    from torchvision import transforms
    from datetime import datetime

    # This is the file path of the dataset, it is assumed to just be within './data/*'
    data_fpath = os_getcwd() + '/data'
    # These are the paths for the training, testing and validation data
    data_train_fpath = data_fpath + '/train' # Used for training the model, these define how it works
    data_test_fpath  = data_fpath + '/test' # Used for testing the accuracy of the model.
    data_valid_fpath = data_fpath + '/valid'

    # Long list of variables.
    options = model.get_options("./options.json")


# Synatic sugar, makes the main function look like the start of the program!
if __name__ == "__main__":
    main()