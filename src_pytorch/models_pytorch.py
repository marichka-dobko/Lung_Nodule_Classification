import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils


class CNNT4(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNT4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(5 * 5 * 16, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(-1, 16 * 5 * 5)  # 16 * 5 * 5
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)
        return out  # F.log_softmax(out, dim=1)  

    
    
class CNNT5(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNT5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(5 * 5 * 16, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(-1, 16 * 5 * 5)  # 16 * 5 * 5
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc_out(out)
        return out  # F.log_softmax(out, dim=1)  

    
class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, kernel_size=(5, 5, 5))
        # self.conv1 = nn.Conv3d(3, 6, kernel_size=(5, 5, 5))   # initial dimensionality
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out   # F.log_softmax(x, dim=1)   # initial: x
    

class CNNT4_3D(nn.Module):
    def __init__(self):
        super(CNNT4_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, kernel_size=(5, 5, 5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.fc_out(out)
        return out   # F.log_softmax(out, dim=1)   


class CNNT5_3D(nn.Module):
    def __init__(self):
        super(CNNT5_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(5, 5, 5))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_out = nn.Linear(50, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16 * 5 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.fc_out(out)
        return out   # F.log_softmax(out, dim=1)   

