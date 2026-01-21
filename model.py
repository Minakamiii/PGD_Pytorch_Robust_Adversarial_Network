"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):  
      super(Model, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)   
      self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)      
      self.fc1 = nn.Linear(7 * 7 * 64, 1024)
      # 输出层
      self.fc2 = nn.Linear(1024, 10)
      self._init_weights()
      
  def _init_weights(self):
      """对应原代码的 _weight_variable 和 _bias_variable"""
      for m in self.modules():
          if isinstance(m, (nn.Conv2d, nn.Linear)):
          # 截断正态分布初始化，标准差为 0.1
              nn.init.trunc_normal_(m.weight, std=0.1, a=-0.2, b=0.2)
            # 偏置初始化为 0.1
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0.1)
  def forward(self, x):
        # 对应 tf.reshape(self.x_input, [-1, 28, 28, 1])
        # PyTorch 期望 [Batch, Channel, Height, Width]
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.shape[1] == 784: # 处理类似 (B, 784) 的输入
             x = x.view(-1, 1, 28, 28)

        # 第一层：卷积 -> ReLU -> 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # 第二层：卷积 -> ReLU -> 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # 展平：对应 tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        x = x.view(-1, 7 * 7 * 64)

        # 全连接层 1 -> ReLU
        x = F.relu(self.fc1(x))

        # 输出层 (对应 self.pre_softmax)
        logits = self.fc2(x)
        
        return logits