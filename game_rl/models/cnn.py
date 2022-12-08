import torch.nn as nn

class MiniCNN(nn.Module):
    """
    Mini CNN structure
    input -> (conv2d + relu)x3 -> flatten -> (dense + relu)x2  -> output
    """
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        c, h, w = input_dim
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=c,out_channels=32,kernel_size=8, stride=4),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, input):
        """
        Forward Block of CNN
        """
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.flatten(input)
        input = self.fc1(input)
        input = self.fc2(input)
        return input


