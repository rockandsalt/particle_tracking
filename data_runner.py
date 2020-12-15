import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from skimage import exposure
from skimage import filters
from scipy import ndimage
import scipy.io

import pims

import dask.array as da
import dask

from os import path
import pims

import h5py

import json 
from tqdm import tqdm

def sum_im(im):
    stack = da.stack(im, axis=0, allow_unknown_chunksizes=True)
    dask_a = stack.sum(axis = 0)
    return dask_a.compute()

def find_min_frame(data):
    min_frame = np.inf 
    for _, f_list in data['data'].items():
        for f_name in f_list:
            vid = pims.open(f_name)
            if(len(vid) < min_frame):
                min_frame = len(vid)
    return min_frame

@pims.pipeline
def binarise_half_way(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return ((min_val+(max_val-min_val)*0.25) <= image).astype(int)

@pims.pipeline
def binarise_half_way_err(image):
    image = image.astype(float)
    min_val = np.min(image)
    max_val = np.max(image)
    return ((min_val+(max_val-min_val)*0.25) <= image).astype(int)

def process_frames(vid, crop, min_frame):
    vid = pims.process.crop(vid[:min_frame], crop)
    vid = binarise_half_way(vid)
    return sum_im(vid)

def process_frames_err(vid, crop, min_frame):
    vid = pims.process.crop(vid[:min_frame], crop)
    vid = binarise_half_way_err(vid)
    return sum_im(vid)

def run_loader(path):
    with open(path) as f:
        data = json.load(f)
    output_path = data['output_path']

    cropping_size = data['crop']
    f = h5py.File(output_path, 'w')

    min_frame = find_min_frame(data)

    for key, f_list in data['data'].items():
        area_im = []
        for f_name in tqdm(f_list):
            try:
                vid = pims.open(f_name)
                area_im.append(process_frames(vid, cropping_size, min_frame))
            except:
                print("error on : {}".format(f_name))
        if len(area_im) > 1:
            f.create_dataset(key, data=np.stack(area_im, axis=0))
        else:
            f.create_dataset(key, data=area_im[0])
        f.flush()
    f.close()

def process_erroneous():
    vid = pims.ImageSequence("D:/ultrasound_2_alex/ss316/new_soundfield/tif/ultrasound_228_3*.tif")
    area = process_frames_err(vid, ((120,180),(0,0)), 20037)
    f = h5py.File('./video/ss_228_run_3.h5', 'w')
    f.create_dataset('data', data=area)
    f.flush()
    f.close()

def save_one_frame(path):
    with open(path) as f:
        data = json.load(f)

    f = h5py.File('./video/one_frame.h5', 'w')
    
    for key, f_list in data['data'].items():
        one_frame = []
        for f_name in tqdm(f_list):
            try:
                vid = pims.open(f_name)
                one_frame.append(vid[0])
            except :
                print("error on : {}".format(f_name))
        
        f.create_dataset(key, data=np.stack(one_frame, axis = 0))
        f.flush()
    
    f.close()

if __name__ == "__main__":
    run_loader('./sidefeed.json')