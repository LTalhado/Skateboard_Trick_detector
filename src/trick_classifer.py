'''
'''
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")

# Parameters
BATCH_SIZE = None
Learning_rate = 1e-3

# normailize data for images  
# data / 255
# Frame size of video [244,244]

import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    # Define the convolutional layer with 3 input channels (RGB)
    self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)

  def forward(self, x):
    # Pass the input data (x) through the convolutional layer
    x = self.conv1(x)
    # You can add more layers here (e.g., pooling, activation)
    # ...
    return x


model = Model()


# loss Function

# Training loop


# Save model
# PATH = ''
# torch.save(model.state_dict(), PATH)