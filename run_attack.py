"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import math
import glob
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from model import Model

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Block 2: 检查点查找函数
def get_latest_checkpoint(model_dir):
    """手动实现 tf.train.latest_checkpoint"""
    checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*.pth'))
    if not checkpoints:
        return None
    # 根据文件名中的步数排序
    def get_step(path):
        return int(os.path.basename(path).split('-')[-1].split('.')[0])
    return max(checkpoints, key=get_step)

def load_model(checkpoint_path, model):
    """手动实现 saver.restore"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 验证模式
    
    
# Block 3: 验证逻辑
def run_attack(checkpoint, x_adv, epsilon):
    # 1. 加载原生 MNIST 测试集用于计算 L-inf 距离
    # 原代码用 input_data，这里用 torchvision
    test_dataset = torchvision.datasets.MNIST('./MNIST_data', train=False, download=True)
    # 转为 [10000, 784] 的 numpy 数组，方便和 x_adv 比较
    x_nat = test_dataset.data.view(10000, -1).numpy() / 255.0
    y_true = test_dataset.targets.numpy()

    # 2. L-inf 约束检查 (保持原样)
    l_inf = np.amax(np.abs(x_nat - x_adv))
    if l_inf > epsilon + 0.0001:
        print('maximum perturbation found: {}'.format(l_inf))
        print('maximum perturbation allowed: {}'.format(epsilon))
        return

    # 3. 初始化模型
    model = Model().to(device)
    load_model(checkpoint, model)

    # 4. 批量推理
    num_eval_examples = 10000
    eval_batch_size = 64
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr = 0
    y_pred = []

    print(f"Running evaluation on {device}...")

    with torch.no_grad(): # 替代 sess.run，关闭梯度以提速
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            # 将 numpy 切片转为 Torch 张量并调整形状
            x_batch = torch.from_numpy(x_adv[bstart:bend, :]).float().to(device)
            # 还原为模型需要的 [B, 1, 28, 28]
            x_batch = x_batch.view(-1, 1, 28, 28)
            
            y_batch = torch.from_numpy(y_true[bstart:bend]).long().to(device)

            # 前向传播
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            
            # 统计正确数
            total_corr += (predictions == y_batch).sum().item()
            y_pred.append(predictions.cpu().numpy())

    accuracy = total_corr / num_eval_examples
    print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
    
    # 保存预测结果
    y_pred = np.concatenate(y_pred, axis=0)
    np.save('pred.npy', y_pred)
    print('Output saved at pred.npy')
    
# Block 4: Main Block
if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    model_dir = config['model_dir']
    checkpoint = get_latest_checkpoint(model_dir)
    x_adv = np.load(config['store_adv_path'])

    # 基础安全检查
    if checkpoint is None:
        print('No checkpoint found')
    elif x_adv.shape != (10000, 784):
        print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
    elif np.amax(x_adv) > 1.0001 or np.amin(x_adv) < -0.0001 or np.isnan(np.amax(x_adv)):
        print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
                np.amin(x_adv), np.amax(x_adv)))
    else:
        run_attack(checkpoint, x_adv, config['epsilon'])
