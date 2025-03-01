from rasterio import features
from shapely.geometry import shape
import rasterio
import pickle
import numpy as np
from rasterio import mask
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from shapely import MultiPolygon, Polygon
from shapely.geometry import box
import json
from PIL import Image
import rioxarray

path = "/home/lu/Desktop/Trabajo/deep-crowns-biobio/data/"
f = open(f"{path}ignitions.json")
ignitions = json.load(f)
dir_list = os.listdir(f"{path}hour_graphs/")
isoc = {}
max_box = [0,0]
centers = {}
bigs = {}
max_x_ = 0
max_y_ = 0
cell_res = 100
for i in tqdm(dir_list[:1000]):
    fire_n = i.split("_")[1].split('.')[0]
    with open(f"{path}hour_graphs/{i}", "rb") as file:
        hr_graph = pickle.load(file)
        if len(hr_graph) > 1:
            with rasterio.open(f"{path}landscape/Input_Geotiff.tif") as f:
                image = f.read(1)
                transform = f.transform
                crs = f.crs
                # use f.nodata if possible; it's not defined on this particular image
                nodata = -9999.0
                dims = f.read(1).shape
                mask = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
                idx = np.unravel_index(ignitions[fire_n] - 1, (1903, 1549))
                #print(image[(idx[0]),(idx[1])])
                #assert(image[(idx[0]),(idx[1])] != nodata)
                mask[(idx[0]),(idx[1])] = True
                for t in hr_graph:
                    for k,j in hr_graph[t][0]:
                        idx = np.unravel_index(k - 1, (1903, 1549))
                        mask[(idx[0]),(idx[1])] = True
                        #assert(image[(idx[0]),(idx[1])] != nodata)
                        idx2 = np.unravel_index(j - 1, (1903, 1549))
                        mask[(idx2[0]),(idx2[1])] = True
                            #assert(image[(idx[0]),(idx[1])] != nodata)
                polygons = []
                for coords, value in features.shapes(mask, transform = transform):
                    # ignore polygons corresponding to nodata
                    if value != 0:
                        # convert geojson to shapely geometry
                        geom = shape(coords)
                        polygons.append(geom)
                # use the feature loop in case you polygon is a multipolygon
                if mask.sum():
                    features_ = [0]
                    multi_p = MultiPolygon(polygons) # add crs using wkt or EPSG to have a .prj file
                    min_x, min_y, max_x, max_y = multi_p.bounds
                    bounding_box = ((max_x - min_x) + 320, (max_y - min_y) + 320)
                    centers[fire_n] = (min_x + (max_x - min_x)/2, min_y + (max_y - min_y)/2)
                    x_min, y_min = rasterio.transform.rowcol(transform, min_x, min_y)
                    x_max, y_max = rasterio.transform.rowcol(transform, max_x, max_y)
                    if x_min - x_max > max_x_:
                        max_x_ = x_min - x_max
                    if y_max - y_min > max_y_:
                        max_y_ = y_max - y_min   

dim_ = 432

if max_y_ > dim_ or max_x_ > dim_:
    print(fire_n)
    print(max_x_, max_y_)
    raise Exception("Box too big!")
                    


bbox = (dim_ * cell_res, dim_ * cell_res)
pad = 80
with rasterio.open(f"{path}landscape/Input_Geotiff.tif") as f:
    limits = f.bounds
    limits_x = (limits[0], limits[2])
    limits_y = (limits[1], limits[3])
    dims = f.read(1).shape
    image = f.read(1)
    transform = f.transform


indices_ = {}
for i in tqdm(centers):
    x, y = centers[i][0], centers[i][1]
    max_x = x + (bbox[0]/2)
    min_x = x - (bbox[0]/2)
    max_y = y + (bbox[1]/2)
    min_y = y - (bbox[1]/2)
    if max_x > limits[2]:
        delta = limits[2] - max_x
        max_x += delta
        min_x += delta
    if max_y > limits[3]:
        delta = limits[3] - max_y
        max_y += delta
        min_y += delta
    if min_x < limits[0]:
        delta = limits[0] - min_x
        max_x += delta
        min_x += delta
    if min_y < limits[1]:
        delta = limits[1] - min_y
        max_y += delta
        min_y += delta
    indices = ((bbox[1]))
    ## CHECK:
    if max_x > limits_x[1] or max_y > limits_y[1] or min_x < limits_x[0] or min_y < limits_y[0]:
        print(max_x, min_x, max_y, min_y)
        raise Exception("Out of bounds!")  
    y_max, x_min = rasterio.transform.rowcol(transform, min_x, min_y)
    y_min, x_max = rasterio.transform.rowcol(transform, max_x, max_y)
    box_x, box_y = (int((max_x - min_x)/ cell_res), int((max_y - min_y)/cell_res))
    x, y = (int((min_x - limits[0]) / cell_res), int((limits[3] - max_y) / cell_res))
    lands = image[y:y+box_y, x:x+box_x]
    indices_[i] = (y, y + box_y, x, x + box_x)
    assert(lands.shape == (dim_, dim_))
    assert(int((max_x - min_x)/ cell_res) == dim_ and int((max_y - min_y)/cell_res))
    with open(f"{path}/hour_graphs/graph_{i}.pkl", "rb") as file:
        hr_graph = pickle.load(file)
    if len(hr_graph) > 1:
        nodata = -9999.0
        mask = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
        idx = np.unravel_index(ignitions[i] - 1, (1903, 1549))
        mask[(idx[0]),(idx[1])] = True
        shape = (int(bbox[1])//cell_res, int(bbox[0]//cell_res))
        mask_ = np.zeros(shape, dtype=np.bool_).astype(np.uint8)
        coords = rasterio.transform.xy(transform, idx[0], idx[1])
        x = int((coords[0] - min_x) / cell_res)
        y = int((max_y - coords[1]) / cell_res)
        mask_[y,x] = True
        temp = 0
        for t in hr_graph:
            while t > temp:
                plt.imsave(f"{path}spreads_400/fire_{i}-{temp}.png", mask_)
                plt.imsave(f"{path}spreads_400/iso_{i}-{temp}.png", iso_)
                temp += 1
            iso = np.zeros(dims, dtype=np.bool_).astype(np.uint8)
            iso_ = np.zeros(shape)
            for k,j in hr_graph[t][0]:
                idx = np.unravel_index(k - 1, (1903, 1549))
                mask[(idx[0]),(idx[1])] = True
                iso[(idx[0]),(idx[1])] = True
                coords = rasterio.transform.xy(transform, idx[0], idx[1])
                x = int((coords[0] - min_x) / cell_res)
                y = int((max_y - coords[1]) / cell_res)
                mask_[y,x] = True
                iso_[y,x] = True
                idx2 = np.unravel_index(j - 1, (1903, 1549))
                mask[(idx2[0]),(idx2[1])] = True
                iso[(idx2[0]),(idx2[1])] = True
                coords = rasterio.transform.xy(transform, idx2[0], idx2[1])
                x = int((coords[0] - min_x) / cell_res)
                y = int((max_y - coords[1]) / cell_res)
                mask_[y,x] = True
                iso_[y,x] = True
            assert(mask_.sum() != 0)
            plt.imsave(f"{path}spreads_400/fire_{i}-{t}.png", mask_)
            plt.imsave(f"{path}spreads_400/iso_{i}-{t}.png", iso_)
            temp = t + 1
with open(f"{path}indices.json", 'w', encoding='utf-8') as f:
    json.dump(indices_, f)

