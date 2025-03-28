import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/lu/Desktop/Trabajo/deep-crowns-biobio/utils/")
# sys.path.append("/home/lucas/deep-crowns-biobio/utils/")
from torch.utils.data import DataLoader
from utils import MyDataset, Normalize, MyDatasetV2
from tqdm import tqdm
from networks.conv_net import Conv_Net
from networks.conv_net_2 import Conv_Net2
from networks.unet import U_Net
from networks.unet_v2 import U_Net_V2
from networks.utils import EarlyStopper
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
import random


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--net', type=str, default="u-net")
parser.add_argument('--batch_size', type=int, default = 1)
parser.add_argument('--workers', type=int, default = 1)
parser.add_argument('--path', type=str, default="./")

# Random experiment number
exp_number = random.randint(0, 1000)
print(f"Experiment number: {exp_number}")   
writer = SummaryWriter(f"runs/experiment{exp_number}")

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
wd = args.weight_decay
network = args.net
batch_size = args.batch_size
workers = args.workers
path = args.path

nets = {
    "conv": Conv_Net,
    "conv-2": Conv_Net2,
    "u-net": U_Net,
    "u-net-2": U_Net_V2,
}
path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio"
# path = "/home/lucas/deep-crowns-biobio"
transform = None
dataset = MyDatasetV2(path + "/data", tform=transform)
generator = torch.Generator().manual_seed(123)
train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, generator=generator, num_workers=workers, pin_memory=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, generator=generator, num_workers=workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, generator=generator, num_workers=workers, pin_memory=True)

net = nets[network]({"cam": False, "path": path + "/src"})
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
net.cuda(0)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
early_stopper = EarlyStopper(patience=5, min_delta=0.01)

accuracy = BinaryAccuracy()
precision = BinaryPrecision()
recall = BinaryRecall()
f1 = BinaryF1Score()

accuracy_val = BinaryAccuracy()
precision_val = BinaryPrecision()
recall_val = BinaryRecall()
f1_val = BinaryF1Score()

if batch_size == 1:
        net.eval()
for epoch in tqdm(range(epochs)):
    n = 0
    # Reset metrics
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    accuracy_val.reset()
    precision_val.reset()
    recall_val.reset()
    f1_val.reset()
    for _, x, y in tqdm(train_loader):
        net.zero_grad()
        pred = net((x[0].cuda(0), x[1].cuda(0)))
        loss = net.train_loss(pred, y.cuda(0))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probs = F.sigmoid(pred).flatten()
            target = y.flatten().cuda(0)
            accuracy.update(probs, target.int())
            precision.update(probs, target.int())
            recall.update(probs, target.int())
            f1.update(probs, target.int())

    for _, x, y in validation_loader:
        pred = net((x[0].cuda(0), x[1].cuda(0)))
        loss = net.validation_loss(pred, y.cuda(0))
        scheduler.step(loss)
        with torch.no_grad():
            probs = F.sigmoid(pred).flatten()
            target = y.flatten().cuda(0)
            accuracy_val.update(probs, target.int())
            precision_val.update(probs, target.int())
            recall_val.update(probs, target.int())
            f1_val.update(probs, target.int())
        
    # if early_stopper.early_stop(net.val_epoch_loss):             
    #   print("Early stoppage at epoch:", epoch)
    #   break

    writer.add_scalar("Loss/train", net.epoch_loss/net.n, epoch)
    writer.add_scalar("Loss/val", net.val_epoch_loss/net.m, epoch)
    writer.add_scalar("Accuracy/train", accuracy.compute().item(), epoch)
    writer.add_scalar("Accuracy/val", accuracy_val.compute().item(), epoch)
    writer.add_scalar("Precision/train", precision.compute().item(), epoch)
    writer.add_scalar("Precision/val", precision_val.compute().item(), epoch)
    writer.add_scalar("Recall/train", recall.compute().item(), epoch)
    writer.add_scalar("Recall/val", recall_val.compute().item(), epoch)
    writer.add_scalar("F1/train", f1.compute().item(), epoch)
    f1_val = f1_val.compute().item()
    writer.add_scalar("F1/val", f1_val, epoch)
    # Save best model if val f1 is lower than previous
    if epoch == 0:
        best_f1 = f1_val
    elif f1_val > best_f1:
        best_f1 = f1_val
        path_ = f"{path}/src/networks/weights/{net.name}_{epochs}.pth"
        torch.save(net.state_dict(), path_)
    net.reset_losses()


net.plot_loss(epochs=epochs)
net.finish(epochs)
    
results = {}
results["accuracy"] = accuracy.compute().item()
results["precision"] = precision.compute().item()
results["recall"] = recall.compute().item()
results["f1"] = f1.compute().item()
with open(f'{path + "/src"}/plots/results_{net.name}_{epochs}.json', 'w') as f:
    json.dump(results, f)

results_val = {}
results_val["accuracy"] = accuracy_val.compute().item()
results_val["precision"] = precision_val.compute().item()
results_val["recall"] = recall_val.compute().item()
results_val["f1"] = f1_val.compute().item()
with open(f'{path + "/src"}/plots/results_val_{net.name}_{epochs}.json', 'w') as f:
    json.dump(results_val, f)
