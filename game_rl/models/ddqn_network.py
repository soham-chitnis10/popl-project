import torch.nn as nn
from .cnn import MiniCNN
import copy 


class DDQN_network(nn.Module):
    """
    mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim, ):
        super().__init__()
        c, h, w = input_dim

        self.online = MiniCNN(input_dim, output_dim)

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        Forward pass of networks
        """
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
            