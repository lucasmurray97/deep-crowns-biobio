import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt
from torchvision.io import read_image
import fiona
import rasterio
import rasterio.mask
import pathlib
import rioxarray
import pandas as pd
import os
from tqdm import tqdm
import json

class MyDatasetV2(torch.utils.data.Dataset):
    """Creates dataset that sampes: (obs_id, landscape + fire_t, isocrone_(t+1)).
    obs_id = an identifier for the observation
    landscape = (fuels, arqueo, canopy bulk density, canopy base height, elevation, flora, paleo, urban)
    fire = bit mask of fire current state
    isocrone = bit mask of fire evolution at time t+1
    Args:
        root (str): directory where data is being stored
        tform (Transform): tranformation to be applied at sampling time.
    """
    def __init__(self, root, tform=None):
        super(MyDatasetV2, self).__init__()
        self.root = root
        # Look for spreading data in /spreads_400/ and generate dict with keys corresponding to the data
        HOME_FOLDER = pathlib.Path(root + "/spreads_400/")
        self.fires = {}
        self.isoc = {}
        self.n = 0
        dir_list = set(list(i.name.split('_')[0].split('-')[0] for i in HOME_FOLDER.iterdir()))
        for item in HOME_FOLDER.iterdir():
            if "fire" in item.name:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.fires.keys():
                    self.fires[number] = [spread_number]
                else:
                    self.fires[number].append(spread_number)
            else:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.isoc.keys():
                    self.isoc[number] = [spread_number]
                else:
                    self.isoc[number].append(spread_number)
        self.n = 0
        self.keys = {}
        for i in self.fires:
            self.fires[i].sort()
            self.isoc[i].sort()
            for j in range(len(self.fires[i])):
                if j == len(self.fires[i]) - 1:
                    break
                else:
                    self.keys[self.n] = (i, j)
                    self.n += 1
        
        # Load full landscape and normalize it
        self.transform = tform
        self.data = None
        with rioxarray.open_rasterio(f"{root}/landscape/Input_Geotiff.tif") as src:
            self.data = src.where(src != -9999.0, -1)
            self.mean = self.data.mean(dim=["x", "y"])
            self.std = self.data.std(dim=["x", "y"])
        self.data = (self.data - self.mean) / self.std
        # Load indices from data generation process
        with open(root + "/indices.json") as f:
            self.indices = json.load(f)
        # Load weathers corresponding to each fire

        self.w_history = pd.read_csv(f'{self.root}/landscape/WeatherHistory.csv', header=None)
        self.weathers = {}
        WEATHER_FOLDER = pathlib.Path(root + "/landscape/Weathers")
        for item in WEATHER_FOLDER.iterdir():
            self.weathers[item.name.split("/Weathers/")[0]] = pd.read_csv(f'{self.root}/landscape/Weathers/' + item.name)
        
        w_dirs = os.listdir(f"{self.root}/landscape/Weathers")
        temps = np.zeros((len(w_dirs), 12))
        wind_speeds = np.zeros((len(w_dirs), 12))
        wind_directions = np.zeros((len(w_dirs), 12))
        hums = np.zeros((len(w_dirs), 12))
        for j, i in enumerate(w_dirs):
            weather = pd.read_csv(f'{self.root}/landscape/Weathers/' + i)
            temps[j] = weather["TM"].values
            wind_speeds[j] = weather["WS"].values
            wind_directions[j] = weather["WD"].values
            hums[j] = weather["RH"].values
        self.mu_t = temps.mean()
        self.std_t = temps.std()
        self.mu_ws = wind_speeds.mean()
        self.std_ws = wind_speeds.std()
        self.mu_wd = wind_directions.mean()
        self.std_wd = wind_directions.std()
        self.mu_hr = hums.mean()
        self.std_hr = hums.std()
        
    def __len__(self):
        return 1000
        # return self.n
    
    def __getitem__(self, i):
        fire_number, spread_number = self.keys[i]
        iso_number = spread_number + 1
        assert(spread_number == iso_number - 1)
        y, y_, x, x_ = self.indices[str(fire_number)]
        # Gets landscape data
        topology = self.data[:,y:y_, x:x_].to_numpy()
        # Gets spread data
        spread = read_image(f"{self.root}/spreads_400/fire_{fire_number}-{spread_number}.png")
        spread = torch.where(spread[1] == 231, 1.0, 0.0)
        isoc = read_image(f"{self.root}/spreads_400/iso_{fire_number}-{iso_number}.png")
        isoc = torch.where(isoc[1] == 231, 1.0, 0.0).unsqueeze(0)
        try:
            input = torch.cat((spread.unsqueeze(0), torch.from_numpy(topology)))
        except:
            print(fire_number, spread_number)
            print(topology.shape)
            print(spread.shape)
            print(isoc.shape)
            print(y - y_, x - x_)
            raise
        input = torch.cat((spread.unsqueeze(0), torch.from_numpy(topology)))
        # Gets weather data
        n_weather = self.w_history.iloc[int(fire_number)-1].values[0].split("Weathers/")[1]
        weather = self.weathers[n_weather]
        scenario_n = spread_number 
        wind_speed = (weather.iloc[scenario_n]["WS"] - self.mu_ws) / self.std_ws
        wind_direction = (weather.iloc[scenario_n]["WD"] - self.mu_wd) / self.std_wd
        temperature = (weather.iloc[scenario_n]["TM"] - self.mu_t) / self.std_t 
        humidity = (weather.iloc[scenario_n]["RH"] - self.mu_hr) / self.std_hr
        weather_tensor = torch.Tensor([wind_speed, wind_direction, temperature, humidity])
        if self.transform:
            input = self.transform(input)
            isoc = self.transform(isoc)
            # weather_tensor = self.transform(weather_tensor)
        return (fire_number, iso_number), (input, weather_tensor), isoc

class MyDataset(torch.utils.data.Dataset):
    """Creates dataset that sampes: (landscape + fire_t, isocrone_(t+1)).
    landscape = (fuels, arqueo, canopy bulk density, canopy base height, elevation, flora, paleo, urban)
    fire = bit mask of fire current state
    isocrone = bit mask of fire evolution at time t+1
    Args:
        root (str): directory where data is being stored
        tform (Transform): tranformation to be applied at sampling time.
    """
    def __init__(self, root, tform=None):
        super(MyDataset, self).__init__()
        self.root = root
        HOME_FOLDER = pathlib.Path(root + "/spreads_400/")
        self.fires = {}
        self.isoc = {}
        self.n = 0
        dir_list = set(list(i.name.split('_')[0].split('-')[0] for i in HOME_FOLDER.iterdir()))
        for item in HOME_FOLDER.iterdir():
            if "fire" in item.name:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.fires.keys():
                    self.fires[number] = [spread_number]
                else:
                    self.fires[number].append(spread_number)
            else:
                number = int(item.name.split("_")[1].split('-')[0])
                spread_number = int(item.name.split("_")[1].split('-')[1].split('.')[0])
                if number not in self.isoc.keys():
                    self.isoc[number] = [spread_number]
                else:
                    self.isoc[number].append(spread_number)
        self.n = 0
        self.keys = {}
        for i in self.fires:
            self.fires[i].sort()
            self.isoc[i].sort()
            for j in range(len(self.fires[i])):
                if j == len(self.fires[i]) - 1:
                    break
                else:
                    self.keys[self.n] = (i, j)
                    self.n += 1
        
        self.transform = tform

        
    def __len__(self):
        return self.n
    
    def __getitem__(self, i):
        fire_number, spread_number = self.keys[i]
        iso_number = spread_number + 1
        assert(spread_number == iso_number - 1)
        file = np.load(f'{self.root}/backgrounds_400/background_{fire_number}.npz')
        topology = np.concatenate([np.expand_dims(file["a1"], axis=0), np.expand_dims(file["a2"], axis=0), np.expand_dims(file["a3"], axis=0)
                                , np.expand_dims(file["a4"], axis=0), np.expand_dims(file["a5"], axis=0), 
                                np.expand_dims(file["a6"], axis=0), np.expand_dims(file["a7"], axis=0), 
                                np.expand_dims(file["a8"], axis=0)])
        spread = read_image(f"{self.root}/spreads_400/fire_{fire_number}-{spread_number}.png")
        spread = torch.where(spread[1] == 231, 1.0, 0.0)
        isoc = read_image(f"{self.root}/spreads_400/iso_{fire_number}-{iso_number}.png")
        isoc = torch.where(isoc[1] == 231, 1.0, 0.0).unsqueeze(0)
        input = torch.cat((spread.unsqueeze(0), torch.from_numpy(topology)))
        w_history = pd.read_csv(f'{self.root}/landscape/WeatherHistory.csv', header=None)
        n_weather = w_history.iloc[int(fire_number)-1].values[0].split("Weathers/")[1]
        weather = pd.read_csv(f'{self.root}/landscape/Weathers/' + n_weather)
        scenario_n = spread_number 
        wind_speed = weather.iloc[scenario_n]["WS"]
        wind_direction = weather.iloc[scenario_n]["WD"]
        weather_tensor = torch.Tensor([wind_speed, wind_direction])
        if self.transform:
            input = self.transform(input)
            isoc = self.transform(isoc)
            weather_tensor = self.transform(weather_tensor)
        return f"{fire_number}-{iso_number}", (input, weather_tensor), isoc
            

        

class Normalize(object):
    """Normalizes image channnel by channel.

    Args:
        image (Tensor): image of dimension (C x H x W), where normalization will be carried out
        independently for the C channels
    """

    def __init__(self, root ="../data"):
        self.root = root
        with rioxarray.open_rasterio(f"{self.root}/landscape/Input_Geotiff.tif") as src:
            self.maxes = src.max(dim=["x", "y"]).values
        w_dirs = os.listdir(f"{self.root}/landscape/Weathers")
        self.max_wd = 0
        self.max_ws = 0
        self.min_wd = 100
        self.min_ws = 100
        for i in w_dirs:
            weather = pd.read_csv(f'{self.root}/landscape/Weathers/' + i)
            for row in weather.iterrows():
                wd = row[1]["WD"]
                ws = row[1]["WS"]
                if wd > self.max_wd:
                    self.max_wd = wd
                if ws > self.max_ws:
                    self.max_ws = ws
                if wd < self.min_wd:
                    self.min_wd = wd
                if ws < self.min_ws:
                    self.min_ws = ws
            

    def __call__(self, image):
        shape = image.shape
        if len(shape) > 2:
            if len(shape) > 3:
                for i in range(shape[0]):
                    image[i] = self(image[i])
            else:
                for i in range(1, shape[0]):
                    image[i].apply_(lambda x: (x + 1)/(self.maxes[i-1] + 1))
            return image
        else:
            if len(shape) > 1:
                for i in range(shape[0]):
                    image[i] = self(image[i])
            else:
                image[0] = (image[0] - self.min_ws) / (self.max_ws - self.min_ws)
                image[1] = (image[1] - self.min_wd) / (self.max_wd - self.min_wd)
            return image
        

if __name__ == "__main__":
    dataset = MyDatasetV2("/home/lu/Desktop/Trabajo/deep-crowns-biobio/data")
    nums, (input, weather), isoc = dataset[0]
    nums, (input, weather), isoc = dataset[1]

    
            