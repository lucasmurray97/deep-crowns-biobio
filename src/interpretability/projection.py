# add path to env vars
import sys
sys.path.append("/home/lu/Desktop/Trabajo/deep-crowns-biobio/src/interpretability/CHE/")
from CHE.main import CHE
import rasterio
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import rioxarray
import umap
import json
import os

path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio"

# Load umap model
file_path = path + "/src/interpretability/umap_model.pkl"
print("File exists:", os.path.exists(file_path))
print("File size:", os.path.getsize(file_path), "bytes")
umap_model = joblib.load(path + "/src/interpretability/umap_model.pkl")

# Load CHE model
with open(path + "/src/interpretability/che_model.pkl", "rb") as f:
    che_loaded = pickle.load(f)

# Load .tiff file
with rasterio.open(path + "/src/interpretability/Input_Geotiff.tif") as src:
    # Read three bands
    band1 = src.read(1)
    band2 = src.read(2)
    band3 = src.read(3)

# Construct dataframe with three bands as columns and their indices as i, j
df = pd.DataFrame({
    'fuel': band1.flatten(),
    'cbh': band2.flatten(),
    'dem': band3.flatten(),  
    'i': np.repeat(np.arange(band1.shape[0]), band1.shape[1]),
    'j': np.tile(np.arange(band1.shape[1]), band1.shape[0])
})

# Drop rows with -9999 in fuels
df = df[df['fuel'] != -9999.0]
# Get dummies of the fuel column
fuels_one_hot = pd.get_dummies(df['fuel'].astype(str)).astype(int)
with open(path + "/src/interpretability/fuels.json") as f:
        fuels = json.load(f)
fuel_str_values = [str(i) for i in fuels]
fuels_one_hot = fuels_one_hot.reindex(columns=fuel_str_values, fill_value=0)
df_ = pd.concat([df[["cbh", "dem"]], fuels_one_hot], axis=1)

data = df_.drop(columns = ['34.0'])
data_ = data.to_numpy()

mean1, mean2 = che_loaded.mean
std1, std2 = che_loaded.std

# standarize first two columns
data_[:,0] = (data_[:,0] - mean1) / std1
data_[:,1] = (data_[:,1] - mean2) / std2

# apply umap
umap_feats = umap_model.transform(data_)
contains = che_loaded.contains(umap_feats)

# add contains as a column of df
df['contains'] = contains

# Get the indices of the points that are inside the convex hull
indices = np.where(contains)[0]
not_indices = np.where(contains == False)[0]

# Get the coordinates of the points that are inside the convex hull
coords = df.iloc[indices][['i', 'j']].values
# Get the coordinates of the points that are outside the convex hull
coords_not = df.iloc[not_indices][['i', 'j']].values

# Add a band to the original raster with the indices of the points that are inside the convex hull
band = np.zeros(band1.shape)
band[coords[:, 0], coords[:, 1]] = 0.5
band[coords_not[:, 0], coords_not[:, 1]] = 1
# Save the band as a new raster
with rasterio.open(path + "/src/interpretability/ewes.tif", 'w', driver='GTiff',
                   height=band.shape[0], width=band.shape[1],
                   count=1, dtype='uint8',
                   crs=src.crs, transform=src.transform) as dst:
    dst.write(band, 1)

# Plot the map side by side with the original raster
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Plot the original raster
band1 = band1.astype(np.float32)
band1 = np.where(band1 != -9999, band1, np.nan)
ax[0].imshow(band1, cmap='Greens')
# Add a colorbar
cbar = plt.colorbar(ax[0].imshow(band1, cmap='Greens'), ax=ax[0])
ax[0].set_title('Original Map')
# Plot the convex hull raster
band = band.astype(np.float32)
# band[band == 0] = np.nan
ax[1].imshow(band, cmap='hot')
ax[1].set_title('EWE pixels')
# Add a colorbar
cbar = plt.colorbar(ax[1].imshow(band, cmap='hot'), ax=ax[1])
plt.savefig(path + "/src/interpretability/ewe.png")



