
import torch
from torch import nn
from torchvision.models import vgg16 as torchvis_vgg16

class VGG16_Pytorch(nn.Module):
  '''
    VGG16 using pre-established pytorch
  '''
  def __init__(self, input_size, output_size, lr=0.01):
    super().__init__()
    
    # Get the VGG
    self.vgg16 = torchvis_vgg16()

    # Since the final layer may not have the amount of classes we have, set it.
    vgg_final_layer = len(self.vgg16.classifier) - 1
    vgg_neurons_to_final = self.vgg16.classifier[vgg_final_layer].in_features
    self.vgg16.classifier[vgg_final_layer] = nn.Linear(vgg_neurons_to_final, output_size)
  
    self.lr = lr

  # The forward step
  def forward(self, X):
    return self.vgg16(X)
  
  # The loss function 
  def loss(self, y_hat, y):
    return nn.CrossEntropyLoss()(y_hat, y)

  # The optimization algorithm
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.lr)
