"""
:author: Cameron Sims
:date: 28/08/2025
:description: This file runs the MLP using a trainer
"""
from q3.VGG16 import VGG16_Pytorch as VGG16
from q1.Trainer import Trainer
import q1.model as mdl

def imshow(img):
    # Imports for showing images.
    from matplotlib import pyplot as plt
    import numpy as np

    img = img / 2 + 0.5
    npimg = img.numpy() # Convert the image to a version that numpy can read
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() # Show the plot

def estimate_labels(model, images):
    # Import for estimating 
    from torch import max as torch_max

    # Estimate the images-labels
    output = model(images)
    return torch_max(output, 1).indices

"""
def estimate_get_percent(labels, labels_estimated):
    # Calculate the accuracy
    label_len = len(labels)
    correct = 0
    i = 0
    while i < label_len:
        # If the estimated label is accurate.
        if labels[i] == labels_estimated[i]:
            correct += 1
        i += 1 # Increment

    # Return the accuracy.
    return float(correct / label_len)
"""

def estimate(model, loader, writer):
    # Imports 
    from torchvision.utils import make_grid as make_grid
    import torch.nn as nn

    #for images, labels in test_loader:
    #data_iterator = iter(loader) # Used to iterate through the data.
    #images, labels = next(data_iterator) # Get the next batch of images and labels.
    correct = 0
    amount = 0
    loss_average = 0

    # Estimate the images-labels
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        labels_estimated = estimate_labels(model, images)
        label_amount = len(labels)
        output_amount = images.shape[0]

        #print('Truth:    ', ''.join(f'{label.item():5d}' for label in labels))
        #print('Estimated:', ''.join(f'{label_estimated.item():5d}' for label_estimated in labels_estimated))

        i = 0
        while i < label_amount:
            if labels[i] == labels_estimated[i]:
                correct += 1
            i += 1

        # Get loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_normal = loss.item() * labels.size(0)
        loss_average += loss_normal

        amount += label_amount
    

    loss_average /= amount

    writer.add_scalar('Loss/valid', loss_average, 0)
    writer.add_scalar('Accuracy/valid', correct/amount, 0)

    #imshow(make_grid(images)) # Show the images.
    return float(correct / amount)
   

def main():
    """
    :description: This is where we start from.
    """
    from os import getcwd as os_getcwd
    from torchvision import transforms
    from datetime import datetime
    import torch

    from torch.utils.tensorboard import SummaryWriter

    # This is the file path of the dataset, it is assumed to just be within './data/*'
    data_fpath = os_getcwd() + '/data'
    # These are the paths for the training, testing and validation data
    data_train_fpath = data_fpath + '/train' # Used for training the model, these define how it works
    data_test_fpath  = data_fpath + '/test' # Used for testing the accuracy of the model.
    data_valid_fpath = data_fpath + '/valid'

    # Long list of variables.
    options = mdl.get_options("./q1/options.json")

    # Transformer, if nessessary: resize and convert to a format we can read!
    transformer = transforms.Compose([
        #transforms.Resize((options['img']['width'], options['img']['height'])), # Resize the image to a common size.
        transforms.ToTensor(), # Convert to a tensor, for PyTorch to read.
        transforms.Normalize((0.5,), (0.5,)) # We have RGB Inputs, hence 3 values.
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # These are the imports for reading from the dataset.
    train_dataset, train_loader = mdl.create_dataset(data_train_fpath, transformer, options)

    # These are for validating the dataset.
    valid_dataset, valid_loader = mdl.create_dataset(data_valid_fpath, transformer, options)

    # These are for testing the dataset.
    test_dataset, test_loader = mdl.create_dataset(data_test_fpath, transformer, options)

    # Create a tensorboard.
    writer = SummaryWriter(log_dir=os_getcwd() + '/logs/q1')

    # We want to create the mdl.
    model = mdl.train(model_class=VGG16, options=options, train_loader=train_loader, writer=writer, device=device)

    # Show what the numbers mean 
    # model_print_classes(train_dataset, test_dataset)

    # Use to test the data, get the accuracy
    accuracy = estimate(model, valid_loader, writer)

    # Use the validator...
    print('Graphing...')
    images, labels = next(iter(train_loader)) 
    writer.add_graph(model, images)

    writer.close()


# Synatic sugar, makes the main function look like the start of the program!
if __name__ == "__main__":
    main()