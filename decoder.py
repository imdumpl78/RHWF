import torch
import torch.nn as nn
from utils import *


class CNN_64(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer10(x)
        return x


class CNN_32(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_32, self).__init__()
        
        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x


class GMA_update(nn.Module):
    def __init__(self, args, sz):
        super().__init__()
        self.args = args
        if sz==32:
            self.cnn = CNN_32(80)
        if sz==64:
            self.cnn = CNN_64(64)
            
    def forward(self, corr_flow):      
        delta_flow = self.cnn(corr_flow)   
        return delta_flow
