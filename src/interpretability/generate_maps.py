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
# path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio"
path = "/home/lucas/deep-crowns-biobio"
sys.path.append(path + "/utils/")
sys.path.append(path + "/src/")
from utils import MyDatasetV2
from tqdm import tqdm
from networks.unet import U_Net
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_distance_mask(binary_masks, max_distance=50):
    """
    binary_masks: np.ndarray or torch.Tensor of shape (B, H, W), 1 for object, 0 for background
    max_distance: max distance in pixels to keep attention
    """
    # Make sure we work with numpy
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.cpu().numpy()

    masks = []
    for binary_mask in binary_masks:  # Loop over batch dimension
        distance = distance_transform_edt(1 - binary_mask)
        keep_mask = (distance <= max_distance).astype(np.float32)
        masks.append(keep_mask)

    masks = np.stack(masks, axis=0)  # (B, H, W)
    return masks  # Still in np.float32

def mask_gradcam_border(gradcam_maps, border_width=5):
    """
    gradcam_maps: torch.Tensor of shape (B, H, W)
    border_width: int, how many pixels to mask from each side
    """
    masked = gradcam_maps.clone()
    masked[:,: , :border_width, :] = 0
    masked[:, :, -border_width:, :] = 0
    masked[:, :, :, :border_width] = 0
    masked[:, :, :, -border_width:] = 0
    return masked


def compute_GC(model, x, layer):
    # Holders
    activations_ = []
    gradients_ = []

    # Hooks
    def forward_hook(module, input, output):
        activations_.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients_.append(grad_out[0])

    # Register hooks
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model.forward((x[0].cuda(), x[1].cuda()))  # shape (B, C, H, W) or (B, num_classes)

    # Zero gradients
    model.zero_grad()

    # Create one-hot target tensor
    one_hot = torch.zeros_like(output).cuda()
    class_index = torch.sigmoid(output) >= 0.5
    one_hot[class_index] = 1

    # Backward pass
    output.backward(gradient=one_hot)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Get activations and gradients
    activations = activations_[0]  # Shape: (B, C, H, W)
    gradients = gradients_[0]      # Shape: (B, C, H, W)

    with torch.no_grad():
        # Global average pooling: per batch
        alpha_k_c = gradients.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)

        # Weighted sum over channels
        L_c_grad_CAM = (alpha_k_c * activations).sum(dim=1, keepdim=True)  # Shape: (B, 1, H, W)

        # Apply ReLU
        L_c_grad_CAM = F.relu(L_c_grad_CAM)

        # Normalize each heatmap separately in the batch
        B, _, H, W = L_c_grad_CAM.shape
        L_c_grad_CAM = L_c_grad_CAM.view(B, -1)
        L_c_grad_CAM -= L_c_grad_CAM.min(dim=1, keepdim=True)[0]
        L_c_grad_CAM /= (L_c_grad_CAM.max(dim=1, keepdim=True)[0] + 1e-8)
        L_c_grad_CAM = L_c_grad_CAM.view(B, 1, H, W)

        # Resize heatmaps to desired size (432x432)
        L_c_grad_CAM = F.interpolate(L_c_grad_CAM, size=(432, 432), mode="bilinear")

    return L_c_grad_CAM.cpu(), torch.sigmoid(output.detach().cpu())

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
        im, output = compute_GC(net, x, net.base_model.decoder2)
        im = mask_gradcam_border(im, border_width=5)
        distance_ = compute_distance_mask(y.numpy(), max_distance=50)
        im *= distance_
        # plot im and y side by side
        for i in range(im.shape[0]):
            fig, ax = plt.subplots(1, 3)
            # increase width of the figure
            fig.set_figwidth(15)
            ax[1].imshow(im[i][0])
            ax[1].set_title("Heatmap")
            ax[2].imshow(output[i][0])
            ax[2].set_title("Next state")
            ax[0].imshow(x_i[i][0])
            ax[0].set_title("Previous state")
        if y.sum() * 0.64 >= 100:
            if n_ewes < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/ewes/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_ewes += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/ewes/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        elif y.sum() * 0.64 > 100 and y.sum() * 0.64 >= 50:
            if n_100 < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/100/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_100 += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/100/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        elif y.sum() * 0.64 > 50 and y.sum() * 0.64 >= 20:
            if n_50 < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/50/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_50 += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/50/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        elif y.sum() * 0.64 > 20 and y.sum() * 0.64 >= 10:
            if n_20 < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/20/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_20 += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/20/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        elif y.sum() * 0.64 > 10 and y.sum() * 0.64 >= 5:
            if n_10 < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/10/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_10 += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/10/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        else:
            if n_5 < 100:
                plt.savefig(f'{path}/src/interpretability/attention_maps_plots/5/{idx[0][i].item()}-{idx[1][i].item()}.png')
                n_5 += 1
            # save im as numpy array
            np.save(f'{path}/src/interpretability/attention_maps/5/{idx[0][i].item()}-{idx[1][i].item()}.npy', im[i])
        plt.close()
        n += 1
