import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

## output size of a convolution layer per side of dim size
def conv2d_size_out(size, kernel_size=5, stride=2):
    # TODO
    """Computes output size of a 2D convolutional layer.

    Args:
        size: An open smalltable.Table instance.
        kernel_size: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        stride: Optional; If require_all_keys is True only
          rows with values set for all keys will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:
    """
    return (size - (kernel_size - 1) - 1) // stride + 1

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class BaseAgent(nn.Module):
    # TODO: refactor names
    """Base definition for an agent module.

    Attributes:
        h: A boolean indicating if we like SPAM or not.
        w: An integer count of the eggs we have laid.
        outputs: 
    """
    
    def __init__(self, h, w, outputs):
        super(BaseAgent, self).__init__()
        self.h = h
        self.w = w
        self.outputs = outputs
        
    def forward(self, x):
        pass

class ReplayMemory(object):
    """Replay memory class for deep-rl tasks.

    Attributes:
        capacity (int): Number of memory instances to be stored.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._memory = []
        self._position = 0
        
    def push(self, *args):
        '''Saves a transition.'''
        if len(self._memory) < self.capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        '''Samples a number batch_size of instances from the memory buffer.'''
        return random.sample(self._memory, batch_size)
    
    def __len__(self):
        return len(self._memory)