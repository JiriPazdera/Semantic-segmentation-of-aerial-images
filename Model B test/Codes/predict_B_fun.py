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


def predict_modelB(image_path, scale):    
    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

        return iou

    def dice_coef(y_true, y_pred, smooth = 1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def soft_dice_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    model_path = "C:/Users/jiric/Documents/VUT/BP/UnetModel.h5"
    model = load_model(model_path, custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})
    model.load_weights("C:/Users/jiric/Documents/VUT/BP/Final_unet_road_weights.h5")

    # image_path = r"C:\Users\jiric\Documents\VUT\BP\OBRAZKY\BIG_FORMAT_RESULTS\MODEL_MASSACHUSETTS\MASS_S3\img.jpg"
    # image_path = r"C:\Users\jiric\Documents\VUT\BP\OBRAZKY\BIG_FORMAT_RESULTS\MODEL_MASSACHUSETTS\LCOVER_S3\img.jpg"
    # image_path = r"C:\Users\jiric\Documents\VUT\BP\OBRAZKY\BIG_FORMAT_RESULTS\CUSTOM\WRTO24.2022.VYSK11.jpg"

    patch_size = 256

    # scale mi udava, kolikat zmensim rozliseni
    # scale = 2

    image_bgr = cv2.imread(image_path)

    image_scaled = cv2.resize(image_bgr, (image_bgr.shape[1] // scale, image_bgr.shape[0] // scale))

    height = image_scaled.shape[0]
    width = image_scaled.shape[1]

    new_height = height - (height % patch_size)
    new_width = width - (width % patch_size)

    image_cropped = image_scaled[:new_height, :new_width, :]

    

    print("Original image shape:", image_bgr.shape)
    # print("SCALE =",scale)
    # print("Scaled image shape:", image_scaled.shape)
    # print("CROPPED IMAGE", image_cropped.shape)

    patches = patchify(image_cropped,(patch_size, patch_size, 3), step=patch_size)
    # print("initial patches:", patches.shape)

    pred_image = np.zeros((image_cropped.shape[0], image_cropped.shape[1],1))

    start_time = time.time()

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):

            # print("Predicting patch:",i,j)

            patch = patches[i,j]

            # print("UNIQUE:", np.unique(patch))

            predicted_patch = model.predict(patch)

            predicted_patch = predicted_patch[0,:,:,:]

            class_matrix = np.where(predicted_patch < 0.1, 1, 0)

            pred_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = class_matrix
    
    end_time = time.time()
    final_time = end_time - start_time

    print("FINAL TIME:", final_time)

    # print("PREDICTION FINISHED")
    # print("PREDICTION SHAPE AND VALUES:",pred_image.shape, np.unique(pred_image))
    print("Predicted image shape:", pred_image.shape)

    return image_bgr, image_cropped, pred_image