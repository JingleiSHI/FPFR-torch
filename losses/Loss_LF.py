"""
We can define our own loss here by modifying 'forward' function, you don't have to do this,
because we can find all basic loss function from torch.nn (see pytorch docs).
e.g. Binary Cross Entropy for DCGAN
"""

import torch
import torch.nn as nn

class Loss_L1(nn.Module):
    """
    you should at least implement '__init__' and 'forward' (name should not be changed) in order
    to be used by pytorch.
    """
    def __init__(self):
        super(Loss_L1,self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, logits, labels):
        """
        forward is an important function for calculating loss here, input parameters can be changed (add or remove)
        by your need for your calculation.
        """
        loss = self.loss(logits, labels)
        return loss