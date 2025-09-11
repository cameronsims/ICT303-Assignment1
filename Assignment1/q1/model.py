"""
:author: Cameron Sims
:date: 09/09/2025
:description: This file trains and interacts with the model
"""
from q1.Trainer import Trainer

def get_options(fname):
    # Allow for JSON input 
    from json import load as json_load

    # Long list of variables.
    options_file = open(fname)
    return json_load(options_file)

def create_dataset(fpath, transformer, options):
    # Cool imports
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    # These are for testing the dataset.
    dataset = ImageFolder(root=fpath, transform=transformer)
    loader = DataLoader(dataset, batch_size=options['loader']['batch_size'], shuffle=options['loader']['shuffle'], num_workers=options['loader']['worker_amount'])
    return dataset, loader

def train(model_class, options, train_loader, writer, device):
    # Width*Height*Amount of Colours
    input_size = options['img']['colours']*options['img']['height']*options['img']['width']

    # We want to create the model.
    model = model_class(input_size=input_size, output_size=options['model']['output_size'], lr=options['model']['learning_rate'])
    trainer = Trainer(n_epochs=options['trainer']['epoch_amount'], batch_amount=options['trainer']['batch_fit_amount'], writer=writer, device=device) # Create the trainer
    trainer.fit(model=model, data=train_loader) # Fit the model, with no data for now.

    # Return our MLP class
    return model

def print_classes(ds1, ds2):
    # Get the dataset idx 
    ds1_names = ds1.class_to_idx
    ds2_names = ds2.class_to_idx

    for name in ds1_names:
        # Print out the classes an the number they're associated with...
        id1 = ds1_names[name]
        #id2 = ds2_names[name]
        print(f'{name:12}|{id1:2}')#|{id2:2%}')

def print_performance(options: dict, total_time: int, accuracy: float):
    formatted_str = str(options['model']['learning_rate']) + ',' + \
                    str(options['trainer']['epoch_amount']) + ',' + str(options['trainer']['batch_fit_amount']) + ',' + \
                    str(options['loader']['batch_size']) + ',' + str(options['loader']['worker_amount']) + ',' + str(options['loader']['shuffle']) + ',' + \
                    str(total_time) + ',' + str(accuracy)
    print(formatted_str)