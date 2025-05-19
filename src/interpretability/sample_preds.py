import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys
path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio"
# path = "/home/lucas/deep-crowns-biobio"
sys.path.append(path + "/utils/")
sys.path.append(path + "/src/")
from utils import MyDatasetV2
from tqdm import tqdm
from networks.unet import U_Net
import numpy as np
from scipy.ndimage import distance_transform_edt


if __name__ == "__main__":
    transform = None
    dataset = MyDatasetV2(path + "/data", tform=transform)
    train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

    net = U_Net({"cam": True, "path": path + "/src"})
    net.load_state_dict(torch.load(path+"/src/networks/weights/U-Net_1000.pth"))
    net.cuda()
    net.eval()

    n = 0
    n_ewes = 0
    n_100 = 0
    n_50 = 0
    n_20 = 0
    n_10 = 0
    n_5 = 0
    for idx, x, y in tqdm(train_loader):
        grad = []
        activation = []
        x_i, x_w = x
        pred = net((x_i.cuda(), x_w.cuda()))
        pred = torch.sigmoid(pred).detach().cpu().numpy()
        # plot im and y side by side
        for i in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 3)
            # increase width of the figure
            fig.set_figwidth(15)
            ax[1].imshow(pred[i][0])
            ax[1].set_title("Prediction")
            ax[2].imshow(y[i][0])
            ax[2].set_title("Ground truth")
            ax[0].imshow(x_i[i][0])
            ax[0].set_title("Previous state")
            if y.sum() * 0.64 >= 100:
                if n_ewes < 100:
                    plt.savefig(f'{path}/src/interpretability/attention_maps_plots/ewes/{idx[0][i].item()}-{idx[1][i].item()}__example.png')
                    n_ewes += 1
            
            if n_ewes == 10:
                break