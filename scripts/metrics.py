import numpy as np
import cv2
import pdb

import matplotlib.pyplot as plt

def dimg(img):
    plt.imshow(img)
    plt.show()


def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_mask *= 1.0 
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou

def calculate_overlapping(img_mask, gt_mask):
    img_mask = cv2.resize(img_mask , (32,32))
    gt_mask = cv2.resize(gt_mask , (32,32))
    _,img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
    _,gt_mask = cv2.threshold(gt_mask,127,255,cv2.THRESH_BINARY)


    img_and = cv2.bitwise_and(img_mask*1.0, gt_mask*1.0)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap

def calculate_datsets_belonging( img , wgan):
    # since in the formulation of wgan
    # the real elements that belongs to dataset
    # are given a label of -1 thus we use the same method 
    # to generate the reward and resize the rewards 
    # to lie between 0 and 1 
    # with 1 being the rewards given the image lies perfectly
    # in the dataset distribution of the object

    img = cv2.resize(img , (128,128))
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    valid = wgan.critic.predict(img)[0][0]

    # chaning the range from 1:-1 to 0:1

    valid = (-valid + 1) / 2
    return valid


