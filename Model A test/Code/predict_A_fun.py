import random
import pickle
import os
import cv2
import numpy as np
import tensorflow
from PIL import Image
from matplotlib import pyplot as plt
from patchify import patchify
from enum import Enum
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import time


def predict_modelA(image_path, scale):
    def jaccard_index(y_true, y_pred):
        return 0

    def iou_coefficient(y_true, y_pred):
        return 0
    
    
    model_dir = 'C:/Users/jiric/Documents/VUT/BP/CODES/'
    model_name = 'final_aerial_segmentation_2022-11-09 22_37_27_640199.hdf5'

    model = load_model(
    model_dir + model_name,
    custom_objects={'iou_coefficient': iou_coefficient, 'jaccard_index': jaccard_index})

    patch_size = 160

    colormap = {
        0: [155, 155, 155],         # Unlabelled
        1: [60, 16, 152],           # Building
        2: [132, 41, 246],          # Land
        3: [110, 193, 228],         # Road
        4: [254, 221, 58],          # Vegetation
        5: [226, 169, 41]           # Water
    }

    image_bgr = cv2.imread(image_path)
    print("Original image shape:", image_bgr.shape)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_scaled = cv2.resize(image_rgb, (image_rgb.shape[1] // scale, image_rgb.shape[0] // scale))
    # print("Scaled image shape:", image_scaled.shape)

    height = image_scaled.shape[0]
    width = image_scaled.shape[1]

    new_height = height - (height % patch_size)
    new_width = width - (width % patch_size)

    image_cropped = image_scaled[: new_height, : new_width, :]
    # print("cropped image", image_cropped.shape)

    patches = patchify(image_cropped,(patch_size, patch_size, 3), step=patch_size)
    # print("initial patches:", patches.shape)

    pred_image = np.zeros_like(image_cropped)

    start_time = time.time()

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):

            # print("predicting patch:",i,j)

            patch = patches[i,j]

            predicted_patch = model.predict(patch)

            predicted_patch = predicted_patch[0,:,:,:]

            class_matrix = np.argmax(predicted_patch, axis=-1)

            predicted_patch_rgb = np.zeros((patch_size, patch_size,3))

            for class_id, color in colormap.items():
                predicted_patch_rgb[class_matrix==class_id]=color

            pred_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = predicted_patch_rgb

    end_time = time.time()
    final_time = end_time - start_time

    print("FINNAL TIME:", final_time)

    print("Predicted image shape:", pred_image.shape)

    return image_rgb, image_cropped, pred_image
