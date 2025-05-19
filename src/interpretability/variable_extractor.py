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
import pandas as pd
import json

path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio"
# path = "/home/lucas/deep-crowns-biobio"
sys.path.append(path + "/utils/")
sys.path.append(path + "/src/")
from utils import MyDatasetV2
from tqdm import tqdm
from networks.unet import U_Net
import numpy as np
from scipy.ndimage import distance_transform_edt

def normalize_map(map):
    """
    Normalize the map to the range [0, 1]
    """
    map = (map - np.min(map)) / (np.max(map) - np.min(map))
    return map



if __name__ == "__main__":
    transform = None
    dataset = MyDatasetV2(path + "/data", tform=transform)
    train_dataset, validation_dataset, test_dataset =torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)


    # Create empty pandas dataframe with columns fuel, cbh, dem for first quartile
    # df_0 = pd.DataFrame(columns=["cbh", "dem"])
    # df_1 = pd.DataFrame(columns=["cbh", "dem"])
    # df_2 = pd.DataFrame(columns=["cbh", "dem"])
    df_3 = pd.DataFrame(columns=["cbh", "dem"])

    for idx, x, y in tqdm(train_loader):
        # read numpy from attention_maps
        fire_n, h = idx
        try:
            attention_maps = np.load(path + f"/src/interpretability/attention_maps/ewes/{fire_n.item()}-{h.item()}.npy")
        except:
            continue
        if len(attention_maps.shape) == 3:
            attention_maps = attention_maps[0]
        normalized_map_ = normalize_map(attention_maps)

        fuels = x[0][0][1] * dataset.std[0].data + dataset.mean[0].data
        cbh = x[0][0][2] * dataset.std[1].data + dataset.mean[1].data
        dem = x[0][0][3] * dataset.std[2].data + dataset.mean[2].data



        quart_3 = (normalized_map_ >= 0.99) & (normalized_map_ <= 1)
        fuels_quart_3 = fuels.cpu().numpy()[quart_3]
        cbh_quart_3 = cbh.cpu().numpy()[quart_3]
        dem_quart_3 = dem.cpu().numpy()[quart_3]

        # add to df_3 dframe
        df_3 = pd.concat([df_3, pd.DataFrame({"fuels": fuels_quart_3, "cbh": cbh_quart_3, "dem": dem_quart_3})], ignore_index=True)

        # Get rid of rows with fuels = -1
        df_3 = df_3[df_3['fuels'] != -1]
    # save dataframes to csv
    # Read fuels.json
    with open(path + "/src/interpretability/fuels.json") as f:
        fuels = json.load(f)
    # turn all values into strings
    fuel_str_values = [str(i) for i in fuels]
    fuels_one_hot_3 = pd.get_dummies(df_3['fuels'].astype(str)).astype(int)
    fuels_one_hot_3 = fuels_one_hot_3.reindex(columns=fuel_str_values, fill_value=0)
    df_3 = pd.concat([df_3[["cbh", "dem"]], fuels_one_hot_3], axis=1)
    df_3.to_csv(path + "/src/interpretability/vars/df_99.csv", index=False)