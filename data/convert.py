import cv2, os
import numpy as np
import sys


datset_name = sys.argv[1]
DIR = './quickdraw/' + datset_name + '/'

SOURCE = DIR + 'r128'
imgs = []
for name in os.listdir( SOURCE ):
    print(name)
    if not (name.endswith( 'bmp' ) or name.endswith( 'png' )):
        continue
    img = cv2.imread( SOURCE + '/' + name, 0 )
    imgs += [ img ]
imgs = np.array( imgs )
imgs = ( imgs.astype( np.float32 ) - 127.5 ) / 127.5
imgs = np.expand_dims( imgs, axis=4 )
np.save( DIR + 'object.npy', imgs )
