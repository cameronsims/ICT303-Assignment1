"""
:author: Cameron Sims
:date: 28/08/2025
:description: This file contains the implementation of a simple Multilayer Perceptron (MLP) and a Trainer class for training the MLP on a dataset.
"""


import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    :description: Multilayer Perceptron (MLP) class.
    """

    def __init__(self, input_size, output_size, lr):
        """
        :param input_size: Amount of inputs
        :param output_size: Amount of outputs
        :param lr: Learning rate of the model
        :description: Initializes the MLP
        """

        # Call our superclass constructor
        super().__init__()

        # Assign the parameters to instance variables, used for later training
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr

        self.neurons_first = 64 # Number of neurons in the first hidden layer
        self.neurons_second = 32 # Number of neurons in the second hidden layer
        
        # Create the network, this is the internal MLP structure.
        self.net = nn.Sequential(
            nn.Flatten(), # Flatten the input, 1->Many
            nn.Linear(input_size, self.neurons_first), # First layer
            nn.ReLU(), # Activation function
            nn.Linear(self.neurons_first, self.neurons_second), # Second layer
            nn.ReLU(), # Activation function
            nn.Linear(self.neurons_second, output_size)  # Output layer, Many->Out
        )

    def forward(self, X):
        """
        :param X: Input data
        :description: Forward pass of the MLP
        :return: The output of the MLP, from the given'X' parameter
        """
        #print('X:', X)
        #print('Shape:', X.shape)
        return self.net(X)
    
    def loss(self, y_hat, y):
        """
        :param y_hat: The predicted output
        :param y: The ground truth
        :description: Calculates the loss of the model, high (1.0) inputs for good performance, low (0.0) inputs for bad performance
        :return: The cost of the model, how well did it do?
        """

        # This is the loss according to the neural network
        fn = nn.CrossEntropyLoss()

        # Call the above loss function, and return the result
        loss_value = fn(y_hat, y)

        #print('Loss:', loss_value)

        # Call the above loss function, and return the result
        return loss_value

    def configure_optimizers(self):
        """
        :description: Optimises the MLP
        :return: Returns an optimiser for the MLP
        """

        #print('Optimising...')

        # Return the Adam optimiser, which is a commonly used optimiser for neural networks
        return torch.optim.Adam(self.parameters(), self.lr)
    