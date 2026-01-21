import robustml
import torch
import torch.nn as nn
import numpy as np
import os
import glob

import model

class Model(robustml.model.Model):
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize the model
        self._model = model.Model().to(self._device)
        self._model.eval()

        # 2. find and load the latest checkpoint
        model_dir = 'models/secret'
        checkpoints = glob.glob(os.path.join(model_dir, 'checkpoint-*.pth'))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
            checkpoint = torch.load(latest_checkpoint, map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
        
        # 3. set up dataset and threat model
        self._dataset = robustml.dataset.MNIST()
        self._threat_model = robustml.threat_model.Linf(epsilon=0.3)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        # data preprocessing
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float().to(self._device)
        else:
            x_tensor = x.to(self._device)

        if x_tensor.dim() == 1: # 784
             x_tensor = x_tensor.view(1, 1, 28, 28)
        elif x_tensor.dim() == 2: # (B, 784)
             x_tensor = x_tensor.view(-1, 1, 28, 28)
        elif x_tensor.dim() == 3: # (1, 28, 28)
             x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self._model(x_tensor)
            prediction = torch.argmax(logits, dim=1)
        
        return prediction.cpu().numpy()[0]

    # Properties to mimic TensorFlow model interface
    @property
    def input(self):
        return None

    @property
    def logits(self):
        return self._model

    @property
    def predictions(self):
        return lambda x: torch.argmax(self._model(x), dim=1)