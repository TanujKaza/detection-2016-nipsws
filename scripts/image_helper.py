from keras.preprocessing import image
import numpy as np
import os
import pdb
import cv2

def get_all_dot_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + image_name
        img = cv2.imread(string )
        img = cv2.resize(img , (600,600))
        images.append(img)
    return images

def load_images_names_in_dot_data_set(path_voc):
    file_path = path_voc
    return os.listdir(path_voc)

def mask_image_with_mean_background(mask_object_found, image):
    new_image = image.copy()
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image