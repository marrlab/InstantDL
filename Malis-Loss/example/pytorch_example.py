### use pytorch as a test example
### pytorch version 1.5.0

import numpy as np
import h5py
import malis
from malis.malis_torch import malis_loss2d


##### Loading data 
file_path_training_data = '...'  #please enter file path to training data here
f=h5py.File(file_path_training_data,'r')  

data_ch = f['train']
seg_gt = f['groundtruth']
data_ch = np.transpose(data_ch,(0,-1,1,2))
seg_gt = np.expand_dims(seg_gt,axis=1)


'''
Unet from:
PyTorch implementation of the U-Net for image semantic segmentation with high quality images
https://github.com/milesial/Pytorch-UNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, 2).to(device)
optimizer = optim.Adam(model.parameters())


x_train_tensor = torch.from_numpy(data_ch).float()
seg_train_tensor = torch.from_numpy(seg_gt).float()
dataset = TensorDataset(x_train_tensor, seg_train_tensor)

lengths = [int(len(data_ch)*0.8), int(len(data_ch)*0.2)]
train_dataset, val_dataset = random_split(dataset, lengths)

train_loader = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)


model.train()
print('Begin training ...')
for epoch in range(1, 6):
    l = []
    val_losses = []
    for x_batch, seg_batch in train_loader:
        
        x_batch = x_batch.to(device)
        seg_batch = seg_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = malis_loss2d(seg_batch,output)
        loss.backward()
        optimizer.step()
        l.append(loss.item())
        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch,loss.item()))
        
    torch.save(model, './results/model-{}.pth'.format(epoch))
        
    with torch.no_grad():
        for x_val, seg_val in val_loader:
            x_val = x_val.to(device)
            seg_val = seg_val.to(device)
            model.eval()
            yhat = model(x_val)
            val_loss = malis_loss(yhat, seg_val)
            val_losses.append(val_loss.item())
    print('------ Train Epoch:{} \t, train_loss: {:.6f} \t, val_loss:{:.6f}'.format(epoch,np.mean(l),np.mean(val_losses)))
    

    
    
