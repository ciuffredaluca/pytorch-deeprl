import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv2d_size_out, BaseAgent

class DQN(BaseAgent):
    """Agent class for DQN algorithm (see https://arxiv.org/abs/1312.5602).

    Args:
        h (int): Height of input image.
        w (int): Width of input image.
        outputs (int): Output model size. 
    """ 

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__(h, w, outputs)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, self.outputs)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))