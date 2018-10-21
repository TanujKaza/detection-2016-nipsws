import turtle
from PIL import Image
import io,os
from numpy import random
import gc

class dot(object):
	def __init__(self):
		self.cv = turtle.Canvas(width=600, height=600)
		self.width = 600
		self.height = 600

	def save(self,file):
		self.cv.pack()
		self.cv.update()
		ps = self.cv.postscript(colormode='gray')
		img = Image.open(io.BytesIO(ps.encode('utf-8')))
		img.save('./dot_without_bg/' + str(file) +'.bmp')
		self.cv.destroy()

	def createDot(self):
		
		# place main body
		radius = random.randint(10,40)
		cx = random.randint(radius,self.width-radius)
		cy = random.randint(radius,self.height-radius)
		x0 = cx - radius 
		y0 = cy - radius 
		x1 = cx + radius 
		y1 = cy + radius 		
		self.cv.create_oval( x0 , y0 , x1 , y1 ,fill='Black' )		

for i in range(1000):
	print(i)
	bus = dot()
	bus.createDot()
	bus.save(i)