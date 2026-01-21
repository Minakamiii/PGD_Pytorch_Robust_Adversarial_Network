"""
Implementation of attack methods (PyTorch Version). Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# PyTorch and NumPy Imports
import torch
import torch.nn as nn
import numpy as np

from model import Model


class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.device = next(model.parameters()).device # 获取模型所在的设备

        if loss_func == 'xent':
            # PyTorch standard CrossEntropyLoss (reduction='sum')
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
        elif loss_func == 'cw':
            # CW Loss (Maximizing logit difference: logit_correct - logit_wrong)
            # Minimize: - (logit_correct - logit_wrong + kappa)
            # We will handle this loss calculation inside the perturb method
            self.criterion = loss_func # Store as string to handle in perturb()
            self.kappa = 50 # Equivalent to the constant added in the TF version
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # NOTE: In PyTorch, the gradient graph is created dynamically during perturb(),
        # so we don't pre-calculate self.grad here like in TensorFlow.

    def cw_loss(self, output, target, kappa=50):
        """
        Custom PyTorch implementation of the C&W loss function.
        Replaces the complex tf.one_hot and logit manipulation.
        """
        # Get the logits corresponding to the true labels (correct_logit)
        # output is (batch, num_classes)
        # target is (batch)
        correct_logit = torch.gather(output, 1, target.unsqueeze(1)).squeeze(1)

        # Get the logits for all classes except the true class (wrong_logit)
        # 1. Create a mask to zero out the true class logit
        mask = torch.ones_like(output, dtype=torch.bool).scatter_(1, target.unsqueeze(1), 0)
        
        # 2. Select the max logit from the rest (1-label_mask)*model.pre_softmax
        wrong_logit = output[mask].view(output.size(0), -1).max(dim=1)[0]
        
        # 3. Maximize: logit_correct - logit_wrong. Minimize: - max(...)
        # The 'wrong_logit' part '- 1e4*label_mask' is handled by masking.
        # The ReLU function is handled by torch.clamp(..., min=0)
        
        # loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        loss = -torch.clamp(correct_logit - wrong_logit + kappa, min=0)
        
        return loss.sum() # Sum loss over the batch

    def perturb(self, x_nat, y, device):
        
        # Move data to the specified device
        x_nat = x_nat.to(device)
        y = y.to(device)
        
        # Initialize perturbation start point
        if self.rand:
            # Replaces np.random.uniform and np.clip
            x = x_nat.clone().detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).to(device)
            x = torch.clamp(x, 0, 1) # ensure valid pixel range
        else:
            x = x_nat.clone().detach()

        # IMPORTANT: Enable gradient tracking on the input tensor 'x'
        x.requires_grad = True

        # --- PGD Iterations (Replacing the loop with sess.run(self.grad)) ---
        self.model.eval() # Set model to evaluation mode (no dropout/batchnorm updates)
        
        for _ in range(self.k):
            # Zero out gradients from previous iteration
            if x.grad is not None:
                x.grad.zero_()
            
            # Forward pass
            output = self.model(x)
            
            # Calculate Loss (Dynamic Graph Building)
            if self.criterion == 'cw':
                loss = self.cw_loss(output, y, self.kappa)
            else:
                # Standard Cross-Entropy
                loss = self.criterion(output, y)

            # Backward pass to get gradients
            loss.backward()

            # Get gradient sign
            grad_sign = x.grad.data.sign()

            # The PGD step
            with torch.no_grad():
                # x += self.a * np.sign(grad)
                x.data += self.a * grad_sign
                
                # Projection Step 1: Stay within L-inf ball (Replaces np.clip(x, x_nat - epsilon, x_nat + epsilon))
                eta = torch.clamp(x.data - x_nat.data, -self.epsilon, self.epsilon)
                x.data = x_nat.data + eta

                # Projection Step 2: Ensure valid pixel range (Replaces np.clip(x, 0, 1))
                x.data = torch.clamp(x.data, 0, 1)

        return x.detach()


if __name__ == '__main__':
    # PyTorch Main Execution Environment
    import json
    import sys
    import math
    import glob
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import TensorDataset, DataLoader
    import os

    with open('config.json') as config_file:
        config = json.load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    
    model_dir = config['model_dir']
    checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*.pth'))
    
    if not checkpoints:
        print('No model found')
        sys.exit()
        
    def get_step(path):
        return int(os.path.basename(path).split('-')[-1].split('.')[0])
        
    model_file = max(checkpoints, key=get_step)
    
    
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from {model_file}')

    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])

    
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True, transform=transform)

    num_eval_examples = config.get('num_eval_examples', len(test_dataset))
    eval_batch_size = config['eval_batch_size']
    
    # Prepare data for batched iteration (similar to tf.Session loop)
    # We use a custom DataLoader setup since the original code manually slices a numpy array
    # PyTorch equivalent is to use a DataLoader on the test set.
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    x_adv = [] # adv accumulator

    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    
    print('Iterating over {} batches'.format(num_batches))

    model.eval() 
    
    for ibatch, (x_batch_tensor, y_batch_tensor) in enumerate(test_loader):
        if ibatch >= num_batches:
            break
            
        print('batch size: {}'.format(len(x_batch_tensor)))

        # x_batch is (B, 1, 28, 28) float [0, 1]
        # y_batch is (B) long (integer labels)
        x_batch_adv_tensor = attack.perturb(x_batch_tensor, y_batch_tensor, device)

        # Convert adversarial tensor back to NumPy array and accumulate
        # Detach from graph, move to CPU, convert to NumPy, then remove channel dim (1, 28, 28) -> (28, 28)
        x_batch_adv_np = x_batch_adv_tensor.cpu().numpy()
        
        # The original code's data (mnist.test.images) was (N, 784), so we flatten and convert to float32
        x_batch_adv_np = x_batch_adv_np.reshape(x_batch_adv_np.shape[0], -1) 

        x_adv.append(x_batch_adv_np)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    
    # Slice to ensure only num_eval_examples are stored, matching the original logic
    x_adv = x_adv[:num_eval_examples] 
    
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))