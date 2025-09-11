"""
:author: Cameron Sims
:date: 09/09/2025
:description: This is the VGG16 implementation
"""
# Importing all dependencies
import torch

from q2.VGG16 import VGG16_CNN as VGG16
import q1.model as mdl

"""
class Trainer:

  def __init__(self, n_epochs = 3):
    self.max_epochs = n_epochs
    return

  # The fitting step
  def fit(self, model, data):

    self.data = data

    # configure the optimizer
    self.optimizer = model.configure_optimizers()
    self.model     = model.to(device)

    for epoch in range(self.max_epochs):
      self.fit_epoch()

    print("Training process has finished")

  def fit_epoch(self):

    current_loss = 0.0

    # iterate over the DataLoader for training data
    # This iteration is over the batches
    # For each batch, it updates the network weeights and computes the loss
    for i, (inputs, target) in enumerate(self.data):
      # Get input aand its corresponding groundtruth output
      inputs, target = inputs.to(device), target.to(device)

      # Clear gradient buffers because we don't want any gradient from previous
      # epoch to carry forward, dont want to cummulate gradients
      self.optimizer.zero_grad()

      # get output from the model, given the inputs
      outputs = self.model(inputs)

      # get loss for the predicted output
      loss = self.model.loss(outputs, target)

      # get gradients w.r.t the parameters of the model
      loss.backward()

      # update the parameters (perform optimization)
      self.optimizer.step()

      # Let's print some statisics
      current_loss += loss.item()
      if i % 500 == 499:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0
"""

def main():
    """
    :description: This is where we start from.
    """
    from os import getcwd as os_getcwd
    from torchvision import transforms
    from datetime import datetime

    from torch.utils.tensorboard import SummaryWriter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

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
        transforms.Resize(224, 224), # Resize the image to a common size.
        transforms.ToTensor(), # Convert to a tensor, for PyTorch to read.
        transforms.Normalize((0.5,), (0.5,)) # We have RGB Inputs, hence 3 values.
    ])

    # Start the performance calculations.
    start = datetime.now()

    # These are the imports for reading from the dataset.
    train_dataset, train_loader = mdl.create_dataset(data_train_fpath, transformer, options)

    # These are for validating the dataset.
    valid_dataset, valid_loader = mdl.create_dataset(data_valid_fpath, transformer, options)

    # These are for testing the dataset.
    test_dataset, test_loader = mdl.create_dataset(data_test_fpath, transformer, options)

    # Create a tensorboard.
    writer = SummaryWriter(log_dir=os_getcwd() + '/logs/q2')

    # We want to create the mdl.
    model = mdl.train(model_class=VGG16, options=options, train_loader=train_loader, writer=writer, device=device)

    # Show what the numbers mean 
    # model_print_classes(train_dataset, test_dataset)

    # Use to test the data, get the accuracy
    #accuracy = estimate(model, valid_loader, writer)

    # This is where the performance calc ends.
    end = datetime.now()
    total_time = (end - start)

    # Print out the performance line
    # mdl.print_performance(options, total_time, accuracy)
    

    # Use the validator...
    print('Graphing...')
    images, labels = next(iter(train_loader)) 
    writer.add_graph(model, images)

    writer.close()

# Synatic sugar, makes the main function look like the start of the program!
if __name__ == "__main__":
    main()