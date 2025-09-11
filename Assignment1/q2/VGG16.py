
import torch
from torch import nn

class VGG16_CNN(nn.Module):
  '''
    VGG16 using a CNN
  '''
  def __init__(self, input_size, output_size, lr=0.01):
    super().__init__()

    # Amount of colours that we use...
    colour_amount = 3
    print(input_size, output_size, lr)

    padding_amount = 1
    stride = 2
    kernel_size = 3

    start_amount = 64
    max_neurals = 4096

    first_layer = 6
    second_layer = 16
    third_layer = 32
    fourth_layer = 64
    fifth_layer = 128


    pooling_func = nn.Tanh

    # Convolutional layers
    self.features = nn.Sequential(
      # Block 1 #
      nn.Conv2d(colour_amount, first_layer, kernel_size=kernel_size, padding=padding_amount), # Input: 3*W*H -> 6*W*H
      pooling_func(),
      nn.Conv2d(first_layer, first_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
      
      # Block 2 #
      nn.Conv2d(first_layer, second_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(),
      nn.Conv2d(second_layer, second_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(), 
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride),

      # Block 3 #
      nn.Conv2d(second_layer, third_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(),
      nn.Conv2d(third_layer, third_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(), 
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride),

      # Block 4 #
      nn.Conv2d(third_layer, fourth_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(),
      nn.Conv2d(fourth_layer, fourth_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(), 
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride),

      # Block 5 #
      nn.Conv2d(fourth_layer, fifth_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(),
      nn.Conv2d(fifth_layer, fifth_layer, kernel_size=kernel_size, padding=padding_amount),
      pooling_func(), 
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    )

    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    # Fully connected layers
    self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, max_neurals),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(max_neurals, max_neurals),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(max_neurals, output_size)
    )
  
    self.lr = lr

  # The forward step
  def forward(self, X):
    X = self.features(X)
    X = self.avgpool(X)
    X = torch.flatten(X, 1)
    X = self.classifier(X)
    return X


  # The loss function 
  def loss(self, y_hat, y):
    return nn.CrossEntropyLoss()(y_hat, y)

  # The optimization algorithm
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr)
