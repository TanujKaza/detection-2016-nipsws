import os
import sys
from PIL import Image
import cv2
import sys
import numpy as np

data_type = sys.argv[1]
inp_size = sys.argv[2]
out_size = sys.argv[3]

def resize(folder, fileName, factor):
    filePath = os.path.join(folder, fileName)
    img = cv2.imread(filePath)
    newIm = cv2.resize(img , (int(resizeShape[0]), int(resizeShape[1])) , interpolation = cv2.INTER_AREA)
    newIm = (newIm > 1) * 255
    # i am saving a copy, you can overrider orginal, or save to other folder
    newFolder = './quickdraw/' + data_type + '/r' + str(out_size)
    newFilePath = os.path.join(newFolder, fileName)
    print(fileName)
    os.makedirs( newFolder, exist_ok=True )
    cv2.imwrite(newFilePath , newIm)

def bulkResize(imageFolder, resizeShape):
    imgExts = ["png", "bmp", "jpg"]
    for path, dirs, files in os.walk(imageFolder):
        for fileName in files:
            ext = fileName[-3:].lower()
            if ext not in imgExts:
                continue
            resize(path, fileName, resizeShape)

if __name__ == "__main__":
    imageFolder= './quickdraw/' + data_type + '/r' + str(inp_size)
    resizeShape = [out_size,out_size]
    bulkResize(imageFolder, resizeShape)
    print("resizing compltete" , inp_size , out_size)