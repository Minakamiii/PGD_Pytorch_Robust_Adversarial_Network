from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
import glob # 用于PyTorch风格的检查点查找

# PyTorch Imports
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import numpy as np

from model import Model
from pgd_attack import LinfPGDAttack

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
if eval_on_cpu:
  device = torch.device("cpu")
else:
  device = torch.device("cuda")
  
transform = transforms.Compose(transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST('./MNIST_data', train=False, 
                                          download=True, transform=transform)
model = Model().to(device)
attack = LinfPGDAttack(model, 
                           config['epsilon'],
                           config['k'],
                           config['a'],
                           config['random_start'],
                           config['loss_func'])
global_step = 0

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

summary_writer = SummaryWriter(eval_dir)
criterion = nn.CrossEntropyLoss(reduction='sum')

# replace tf.train.latest_checkpoint(model_dir)
def get_checkpoint_path(model_dir):
  checkpoints = glob.glob(os.path.join(model_dir,'checkpoint-*.pth'))
  
  if not checkpoints:
    return None, 0
  
  def get_step(path):
    base_name = os.path.basename(path)
    last_part = base_name.split('-')[-1]
    step_str = last_part.split('.')[0]
    return int(step_str)
  
  latest_checkpoint = max(checkpoints, key =get_step)
  latest_step = get_step(latest_checkpoint)
  
  return latest_checkpoint,latest_step 

# replace saver.restore(sess, filename)
def load_checkpoint(filename,model):
  checkpoint = torch.load(filename, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict']) 
  
  return checkpoint.get('global_step',0)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  global global_step
  try:
    global_step = load_checkpoint(filename,model)
  except Exception as e:
    print("Error loading the model from checkpoint: ", e)
    return
  model.eval()
  # Iterate over the samples batch-by-batch
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_xent_nat = 0.
  total_xent_adv = 0.
  total_corr_nat = 0
  total_corr_adv = 0
  
  
  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)
    x_batch_np, y_batch_np = test_dataset.data[bstart:bend], test_dataset.targets[bstart:bend]
    x_batch = x_batch_np.float().unsqueeze(1) / 255.0 
    y_batch = y_batch_np
          
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    with torch.no_grad():
      nat_output = model(x_batch)
      cur_xent_nat = criterion(nat_output, y_batch).item()
      nat_pred = nat_output.argmax(dim=1)
      cur_corr_nat = nat_pred.eq(y_batch).sum().item()
    
    x_batch_adv = attack.perturb(x_batch, y_batch, device) #调用 attack.perturb(x_batch, y_batch, sess) 替换为 PyTorch 兼容的调用
    adv_output = model(x_batch_adv)
    cur_xent_adv = criterion(adv_output, y_batch).item()
    adv_pred = adv_output.argmax(dim=1)
    cur_corr_adv = adv_pred.eq(y_batch).sum().item()
    
    total_corr_adv += cur_corr_adv
    total_xent_nat += cur_xent_nat
    total_xent_adv += cur_xent_adv
    total_corr_nat += cur_corr_nat
  
  avg_xent_nat = total_xent_nat / num_eval_examples
  avg_xent_adv = total_xent_adv / num_eval_examples
  acc_nat = total_corr_nat / num_eval_examples
  acc_adv = total_corr_adv / num_eval_examples
  
  summary_writer.add_scalar('xent adv eval', avg_xent_adv, global_step)
  summary_writer.add_scalar('xent adv', avg_xent_adv, global_step)
  summary_writer.add_scalar('xent nat', avg_xent_nat, global_step)
  summary_writer.add_scalar('accuracy adv eval', acc_adv, global_step)
  summary_writer.add_scalar('accuracy adv', acc_adv, global_step)
  summary_writer.add_scalar('accuracy nat', acc_nat, global_step)
  summary_writer.flush()

    # 打印结果
  print('natural: {:.2f}%'.format(100 * acc_nat))
  print('adversarial: {:.2f}%'.format(100 * acc_adv))
  print('avg nat loss: {:.4f}'.format(avg_xent_nat))
  print('avg adv loss: {:.4f}'.format(avg_xent_adv))
'''
# Infinite eval loop
while True:
  cur_checkpoint, cur_step = get_checkpoint_path(model_dir)
  
  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
'''
checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*.pth'))
    
def get_step(path):
    return int(os.path.basename(path).split('-')[-1].split('.')[0])
checkpoints.sort(key=get_step)

for ckpt in checkpoints:
    print(f'\nEvaluating {ckpt}...')
    evaluate_checkpoint(ckpt)
