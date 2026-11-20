import torch
from torch import nn

class Model_Custom(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.dropout(p=0.2)
    )
        self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.dropout(p=0.2)
    )
        self.classifer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=256*14*14, out_features=output_shape)
    )
    def forward(self, x:torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifer(x)
        return x