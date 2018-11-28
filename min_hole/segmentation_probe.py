import sys, os
import csv
from tkinter import *
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import pdb

maskpath = 'data/Mask2/00001.png'
impath = 'data/Mask/00001.png'
impath = 'data/training/00001.jpg'
IMG_SCALE_A = 0.9
LABEL_FILE = 'object150_info.csv'

class UI(Tk):
	def __init__(self):
		super(UI, self).__init__()
		self.bind('<Key>', onkeypress)
		self.img  = None
		self.maskdata = None # numpy array
		self.init_components()
		self.load(impath, maskpath)

	def init_components(self):
		# Canvas
		self.canvas_frame = Frame(self)
		self.canvas_frame.grid(row=0, column=0, rowspan=2)
		self.canvas_frame_a = Frame(self.canvas_frame)
		self.canvas_frame_b = Frame(self.canvas_frame)
		self.canvas_frame_a.grid(row=0, column=0, sticky="nsew")
		self.canvas_frame_b.grid(row=0, column=0, sticky="nsew")
		self.canvas_a0 = Canvas(self.canvas_frame_a)
		self.canvas_b0 = Canvas(self.canvas_frame_b)
		self.canvas_b1 = Canvas(self.canvas_frame_b)
		self.canvas_a0.grid(row=0, column=0)
		self.canvas_b0.grid(row=0, column=0)
		self.canvas_b1.grid(row=0, column=1)
		def init_canvas(canvas):
			canvas.bind('<Motion>', on_canvas_motion)
			canvas.oval_id = None
		init_canvas(self.canvas_a0)
		init_canvas(self.canvas_b0)
		init_canvas(self.canvas_b1)
		# Controls
		ctrl = Frame(self)
		ctrl.grid(row=0, column=1)
		self.display_mode = IntVar()
		for i, lbl in enumerate(('One', 'Two')):
			rb = Radiobutton(ctrl, text=lbl, variable=self.display_mode, value=i)
			rb.pack(anchor=W)
			rb.bind('<Button-1>', onclickmode)
		# Display
		disp = Frame(self)
		disp.grid(row=1, column=1)
		self.label_x   = Label(disp, width=15)
		self.label_y   = Label(disp, width=15)
		self.label_cat = Label(disp, width=15)
		self.set_xy(-1,-1)
		self.label_x.grid(row=0, column=0)
		self.label_y.grid(row=1, column=0)
		self.label_cat.grid(row=2, column=0)

	def load(self, impath, maskpath):
		self.img        = Image.open(impath)
		dim = (self.img.width, self.img.height)
		self.maskdata   = np.array(Image.open(maskpath).getdata()).reshape(dim[1], dim[0])
		# Canvas A0
		self.load_pcolor( self.canvas_a0, np.flip(self.maskdata,0), IMG_SCALE_A )
		# Canvas B0
		self.load_pcolor( self.canvas_b0, np.flip(self.maskdata,0), IMG_SCALE_A*.5 )
		self.load_rgb( self.canvas_b1, self.img, IMG_SCALE_A*.5 )

	# data should be numpy array with segmentation classes as values
	def load_pcolor(self, canvas, data, scale):
		canvas.scaling_factor = scale
		dpi = 100
		fig = plt.figure(figsize=(data.shape[1]*scale/dpi, data.shape[0]*scale/dpi), dpi=dpi)
		ax = fig.add_axes([0,0,1,1])
		ax.set_axis_off()
		plt.pcolormesh( data, cmap='hsv' )
		agg = FigureCanvasAgg(fig)
		agg.draw()
		x, y, w, h = [int(x) for x in fig.bbox.bounds]
		canvas.img = PhotoImage(master=canvas, width=w, height=h)
		canvas.create_image( w/2, h/2, image=canvas.img )
		canvas.config( width=w, height=h )
		tkagg.blit( canvas.img, agg.get_renderer()._renderer, colormode=2 )

	def load_rgb(self, canvas, img, scale):
		canvas.scaling_factor = scale
		img_id = canvas.create_image(0, 0, anchor='nw')
		dim = (img.width, img.height)
		scaled_dim = tuple([round(x*scale) for x in dim])
		scaled_img = img.resize(scaled_dim, Image.ANTIALIAS)
		canvas.img = ImageTk.PhotoImage(scaled_img)
		canvas.config(width=scaled_dim[0], height=scaled_dim[1])
		canvas.itemconfig(img_id, image=canvas.img)

	# Inputs should be unscaled (corresponding to original image, not scaled image)
	def set_xy(self, x, y):
		self.label_x.config(text='X: %4d' % x)
		self.label_y.config(text='Y: %4d' % y)
		if self.maskdata is not None:
			self.cat = self.maskdata[y,x]
		else:
			self.cat = -1
		label = CAT_MAP[self.cat].replace(';','\n') if self.cat >= 0 else ''
		self.label_cat.config(text='C: %4d\n%s' % (self.cat, label))

	def on_canvas_motion(self, evt):
		canvas = evt.widget
		x = round(evt.x / canvas.scaling_factor)
		y = round(evt.y / canvas.scaling_factor)
		if canvas == self.canvas_b0:
			set_oval(self.canvas_b1, evt.x, evt.y)
		elif canvas == self.canvas_b1:
			set_oval(self.canvas_b0, evt.x, evt.y)
		self.set_xy(x,y)

def set_oval(canvas, x, y):
	if canvas.oval_id is not None: canvas.delete(canvas.oval_id)
	canvas.oval_id = canvas.create_oval(x-2, y-2, x+2, y+2, fill='#00ff00', outline='#ff00ff')

def onkeypress(evt):
	if evt.char.lower() == 'q':
		ui.destroy()
	elif evt.char.lower() == 'c':
		ui.canvas.delete(ALL)
	print(evt.char)

def on_canvas_motion(evt):
	ui.on_canvas_motion(evt)

def onclickmode(evt):
	val = evt.widget.config('value')[4]
	if val == 0:
		ui.canvas_frame_a.tkraise()
	else:
		ui.canvas_frame_b.tkraise()

def make_label_map():
	with open(LABEL_FILE, 'r') as f:
		reader = csv.reader(f)
		ls = list(reader)
	dic = {}
	for row in ls[1:]:
		dic[ int(row[0])-1 ] = row[-1]
	return dic

CAT_MAP = make_label_map()
ui = UI()
mainloop()