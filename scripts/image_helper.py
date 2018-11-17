from keras.preprocessing import image
import numpy as np
import os
import pdb
import cv2

def get_all_dot_images(image_names, img_path):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = img_path + image_name
        img = cv2.imread(string , 0 )
        img = cv2.resize(img , (128,128))
        images.append(img)
    return images

def load_images_names_in_dot_data_set(img_path):
    file_path = img_path
    return os.listdir(img_path)

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