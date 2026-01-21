from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Model 
from pgd_attack import LinfPGDAttack 

with open('config.json') as config_file:
    config = json.load(config_file)


# Setting up device and training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']


# Setting up the data and the model
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
global_step = 0 
model = Model().to(device)

# Setting up the optimizer
criterion = nn.CrossEntropyLoss(reduction='sum') 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

shutil.copy('config.json', model_dir)

summary_writer = SummaryWriter(model_dir)

# Main training loop
print(f"Starting training on device: {device}")
training_time = 0.0

# Initialize model to training state
model.train() 

for epoch in range(20):
    # Main training loop
    for i, (x_batch, y_batch) in enumerate(train_loader):
        
        if global_step >= max_num_training_steps:
            break
      
        global_step += 1
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # --- Compute Adversarial Perturbations ---
        model.eval() # Set model to eval mode during attack generation
        start = timer()

        x_batch_adv = attack.perturb(x_batch, y_batch, device) 
        end = timer()
        training_time += end - start

        # Equivalent placeholders for feed_dict (for clarity)
        nat_dict = {'x_input': x_batch, 'y_input': y_batch}
        adv_dict = {'x_input': x_batch_adv, 'y_input': y_batch}

        # Output to stdout
        if global_step % num_output_steps == 0:
            model.eval() 
            with torch.no_grad():
                # Natural Accuracy 
                nat_output = model(x_batch)
                nat_pred = nat_output.argmax(dim=1)
                nat_acc = nat_pred.eq(y_batch).sum().item() / batch_size

                # Adversarial Accuracy 
                adv_output_metric = model(x_batch_adv)
                adv_pred = adv_output_metric.argmax(dim=1)
                adv_acc = adv_pred.eq(y_batch).sum().item() / batch_size

            print('Step {}:    ({})'.format(global_step, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print('training nat accuracy {:.4f}%'.format(nat_acc * 100))
            print('training adv accuracy {:.4f}%'.format(adv_acc * 100))
            
            if global_step != 0:
                print('    {:.2f} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0
            
            model.train() # Set model back to train mode

        # --- Tensorboard summaries ---
        if global_step % num_summary_steps == 0:
            model.eval()
            with torch.no_grad():
                adv_output = model(x_batch_adv)
                loss_adv = criterion(adv_output, y_batch) # Summed loss
                mean_xent_adv = loss_adv.item() / batch_size # Equivalent to model.xent / batch_size
                
                adv_pred = adv_output.argmax(dim=1)
                adv_acc_summary = adv_pred.eq(y_batch).sum().item() / batch_size

            # Log summaries
            summary_writer.add_scalar('accuracy adv train', adv_acc_summary, global_step)
            summary_writer.add_scalar('accuracy adv', adv_acc_summary, global_step)
            summary_writer.add_scalar('xent adv train', mean_xent_adv, global_step)
            summary_writer.add_scalar('xent adv', mean_xent_adv, global_step)
            model.train()


        # --- Write a checkpoint ---
        if global_step % num_checkpoint_steps == 0:
            checkpoint_path = os.path.join(model_dir, 'checkpoint')
            # Saving model and optimizer state dictionaries
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{checkpoint_path}-{global_step}.pth")
            print(f"Saved checkpoint at step {global_step}")

        # --- Actual training step ---
        start = timer()
        optimizer.zero_grad()
        
        # Forward pass on adversarial batch
        adv_output = model(x_batch_adv)
        loss = criterion(adv_output, y_batch) # Summed loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        end = timer()
        training_time += end - start
        
    if global_step >= max_num_training_steps:
        break

print("Training finished.")
summary_writer.close() 