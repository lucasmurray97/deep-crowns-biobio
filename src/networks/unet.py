import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys
from networks.utils import FocalLoss, DiceLoss, ComboLoss
sys.path.append("../utils/")
from torch.utils.data import DataLoader
from utils import MyDatasetV2, Normalize
from tqdm import tqdm

class U_Net(nn.Module):
    def __init__(self, params = {}):
        super(U_Net, self).__init__()
        self.name = "U-Net"
        self.cam = params["cam"]
        self.path = params["path"]
        self.base_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=4, out_channels=1, init_features=128)
        self.fcw_1 = nn.Linear(in_features=4, out_features=81)
        self.bn1 = nn.BatchNorm1d(81)
        self.upconv_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(3,3))
        self.conv1 = nn.Conv2d(in_channels=2049, out_channels=2048, kernel_size=(1,1), stride=(1,1))
        self.bn2 = nn.BatchNorm2d(2048)
        # Loss func
        self.loss = []
        self.epoch_loss = 0
        self.val_loss = []
        self.val_epoch_loss = 0
        self.n = 0
        self.m = 0
        # For gradcam
        self.gradients = None
        self.loss_func = ComboLoss(alpha=0.75, gamma=1.5)

    def forward(self, x):
        x_i, x_w = x
        # Matrix processing
        enc1 = self.base_model.encoder1(x_i.float())
        enc2 = self.base_model.encoder2(self.base_model.pool1(enc1))
        enc3 = self.base_model.encoder3(self.base_model.pool2(enc2))
        enc4 = self.base_model.encoder4(self.base_model.pool3(enc3))
        bottleneck = self.base_model.bottleneck(self.base_model.pool4(enc4))
        # Vector and matrix combination:
        x_w = F.relu(self.bn1(self.fcw_1(x_w)))
        x_w = x_w.view(-1, 1, 9, 9)
        x_w = self.upconv_1(x_w)
        combination = torch.cat((bottleneck, x_w), dim = 1)
        h = self.conv1(combination)
        # For gradcam
        if self.cam:
            h.register_hook(self.activations_hook)
        x_comb = F.relu(self.bn2(h))
        dec4 = self.base_model.upconv4(x_comb)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.base_model.decoder4(dec4)
        dec3 = self.base_model.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.base_model.decoder3(dec3)
        dec2 = self.base_model.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.base_model.decoder2(dec2)
        dec1 = self.base_model.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.base_model.decoder1(dec1)
        out = self.base_model.conv(dec1)
        return out

    def train_loss(self, x, y):
        loss = self.loss_func(x.view(-1, 432, 432), y.view(-1, 432, 432))
        self.epoch_loss += loss.item()
        self.n += 1
        return loss
    
    def validation_loss(self, x, y):
        val_loss = self.loss_func(x.view(-1, 432, 432), y.view(-1, 432, 432))
        self.val_epoch_loss += val_loss.item()
        self.m += 1
        return val_loss
    
    def reset_losses(self):
        self.loss.append(self.epoch_loss/self.n)
        self.val_loss.append(self.val_epoch_loss/self.m)
        self.epoch_loss = 0
        self.val_epoch_loss = 0
        self.n = 0
        self.m = 0
        
    def plot_loss(self, epochs):
        self.to("cpu")
        plt.ion()
        fig = plt.figure()
        plt.plot(self.loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.path}/plots/Losses_{self.name}_{epochs}.png")

    def finish(self, epochs):
        self.plot_loss(epochs)
        path_ = f"{self.path}/networks/weights/{self.name}_{epochs}.pth"
        torch.save(self.state_dict(), path_)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        x_i, x_w = x
        enc1 = self.base_model.encoder1(x_i.float())
        #print(enc1.shape)
        enc2 = self.base_model.encoder2(self.base_model.pool1(enc1))
        #print(enc2.shape)
        enc3 = self.base_model.encoder3(self.base_model.pool2(enc2))
        #print(enc3.shape)
        enc4 = self.base_model.encoder4(self.base_model.pool3(enc3))
        #print(enc4.shape)
        bottleneck = self.base_model.bottleneck(self.base_model.pool4(enc4))
        # Vector and matrix combination:
        x_w = F.relu(self.bn1(self.fcw_1(x_w)))
        x_w = x_w.view(-1, 1, 4, 4)
        #print(x_w.shape)
        x_w = self.upconv_1(x_w)
        #print(x_w.shape)
        combination = torch.cat((bottleneck, x_w), dim = 1)
        #print(combination.shape)
        h = self.conv1(combination)
        return h