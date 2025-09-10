"""
:author: Cameron Sims
:date: 28/08/2025
:description: Trains a MLP model.
"""

class Trainer:
    """
    :description: Trains the MLP
    """

    def __init__(self, n_epochs=3, batch_amount=10, writer=None):
        """
        :param n_epochs: Number of epochs to train for
        :param batch_amount: Number of batches, used to help with fitting
        :description: Creates the trainer class, for use with training the MLP
        """

        self.max_epochs = n_epochs # Set the maximum number of epochs
        self.batch_amount = batch_amount    # The amount of mini-batches
        self.writer = writer

        Trainer.fit_epoch.epoch = 0

    def fit(self, model, data):
        """
        :param model: The MLP model to train
        :param data: The data that we are training with
        :description: The fitting step
        """        
        self.data = data    # Make class aware of the data that we're using.

        # Configure the optimizer using the model's method
        self.optimiser = model.configure_optimizers()   # Seppo spelling because Python
        self.model = model

        # Tell the user we have started training...
        print("Training process has started...")

        # For each epoch, we will fit.
        # Note: I hate using for range(...) in Python, it gives terrible performance with higher values.
        i = 0
        while i < self.max_epochs:
            self.fit_epoch() # Fitting step.
            i += 1

        print("Training process has finished!")

    def fit_epoch(self):
        """
        :description: Fit for the current epoch
        """
        from torch import max as torch_max

        current_loss = 0.0 # The cost, have we done well this epoch?
        Trainer.fit_epoch.epoch += 1
        correct = 0
        total = 0
        
        # For the training data, we will iterate and test the network weights
        # The loss will then be created, compared and see if we need to adjust after mini batching.
        max_batch_remainder = self.batch_amount - 1
        for i, current_data in enumerate(self.data):
            # For each input...
            inputs, target = current_data # Get the inputs, and it's ground truth.
            self.optimiser.zero_grad() # Set the optimiser, removing the gradients.

            # Get important values, from this model given the inputs.
            outputs = self.model(inputs) # What are the outputs?

            # Get the accuracy of the training data...
            index, predicted = torch_max(outputs, 1)
            correct = (predicted == target).sum().item()
            total = target.size(0)


            loss = self.model.loss(outputs, target) # What is the loss from this input-output?

            loss.backward() # Get the gradients of the model.
            self.optimiser.step() # Perform a new step.

            # Add to the writer
            self.writer.add_scalar('Loss/train', current_loss, Trainer.fit_epoch.epoch)
            self.writer.add_scalar('Accuracy/train', correct/total, Trainer.fit_epoch.epoch)

            # Do some mini-batch statistics printing.
            current_loss += loss.item()
            if i % self.batch_amount == max_batch_remainder:
                # Variables to make next line look pretty.
                batch_number = i + 1
                batch_percent = current_loss / self.batch_amount

                # Alert user of the new loss.
                print('Loss after mini-batch %d: %.3f' % (batch_number, batch_percent))

                current_loss = 0.0 # Reset the loss after the elapsed amount of batches.
