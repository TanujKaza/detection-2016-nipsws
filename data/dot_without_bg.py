import turtle
from PIL import Image
import io,os
from numpy import random
import gc
import numpy as np
import pdb
import cv2

mask_info = []
size = 128

def save_bounding_box_info():
	bb_info = np.array(mask_info)
	np.save('bb_info' , bb_info)


class dot(object):
	def __init__(self):
		self.cv = turtle.Canvas(width=128, height=128)
		self.width = 128
		self.height = 128
		# this will tell us where is the dot located in the image

	def save(self,file,path):
		self.cv.pack()
		self.cv.update()
		ps = self.cv.postscript(colormode='gray')
		img = Image.open(io.BytesIO(ps.encode('utf-8')))
		img = np.array(img)
		img = cv2.resize(img , (128,128))
		os.makedirs('./dot_without_bg/' + path + '/' , exist_ok=True)
		cv2.imwrite('./dot_without_bg/' + path + '/' + str(file) +'.bmp' , img)
		self.cv.destroy()

	def createDot(self):
		
		# place main body
		radius = 10
		cx = random.randint(radius,self.width-radius)
		cy = random.randint(radius,self.height-radius)
		x0 = cx - radius 
		y0 = cy - radius 
		x1 = cx + radius 
		y1 = cy + radius 		
		self.cv.create_oval( x0 , y0 , x1 , y1 ,fill='Black' )		
		mask_info.append(np.array([x0, y0 , x1 , y1]))
	

for i in range(1000):
	print(i)
	obj = dot()
	obj.createDot()
	if i < 700:
		obj.save(i , 'train')
	else:
		obj.save(i , 'test')

save_bounding_box_info()