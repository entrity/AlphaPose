import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from options import opts

# Constants for specifying location of data files
# sshfs y6: ~/mnt
BASE_DIR = os.path.expanduser('~/mnt')
IMG_DIR  = os.path.join(BASE_DIR, 'first_third_person/pose_data/training_data/training/training')
MASK_DIR = os.path.join(BASE_DIR, 'first_third_person/Shift-Net_pytorch/datasets/Paris_StreetView_Dataset/Mask2')

MASKED_CATEGORIES = [12, 19, 31, 32, 74, 75,  98, 147, 141, 143]
# MASKED_CATEGORIES += [0]
# MASKED_CATEGORIES += [3]
# MASKED_CATEGORIES += [10]
# MASKED_CATEGORIES += [15]

class Example(object):
	def __init__(self, impath):
		self.impath = impath
		# Load image data and mask data
		self.bname = bname = os.path.basename(self.impath)
		self.name, self.ext = os.path.splitext(self.bname)
		self.mask_path = os.path.join(MASK_DIR, '%s.png' % self.name)
		img  = Image.open(self.impath)
		mask = Image.open(self.mask_path)
		# Build a masked image
		self.maskdata = np.array(mask.getdata(), np.uint8).reshape( mask.height, mask.width )
		self.imdata = np.array(img.getdata(), np.uint8).reshape(img.height, img.width, 3)

	def binmask(self):
		_binmask = np.ones_like( self.maskdata, np.uint8 )
		for category in MASKED_CATEGORIES:
			xmask, ymask = np.where( self.maskdata == category )
			_binmask[xmask, ymask] -= 1
		assert _binmask.min() == 0, _binmask.min()
		return _binmask

	def mask(self, binmask):
		xmask, ymask = np.where( binmask == 0 )
		imdata = self.imdata.copy()
		imdata[xmask, ymask, :] = 0
		return imdata

	def masked(self):
		return self.mask(self.binmask())


class Compiler(object):
	def init_totals(self, imdata, maskdata):
		self.counts = np.zeros_like( maskdata, np.uint32 )
		self.sums   = np.zeros_like( imdata,   np.uint32 )
		self.maxes  = np.zeros_like( imdata,   np.uint8 )
		self.mins   = np.ones_like(  imdata,   np.uint8 ) * 255

	# Given counts and sums from before, make new counts, sums, maxes, mins
	def recompile(self):
		(old_counts, old_sums, old_maxes, old_mins) = load()
		sources = get_sources()
		mean = old_sums / np.expand_dims( old_counts, 2 )

		for i, im_path in enumerate(sources):
			if opts.n and i == opts.n: break

			ex        = Example(im_path)
			imdata   = ex.imdata
			maskdata = ex.maskdata
			print('recompile %6d/%d %s' % (i, len(sources), ex.bname))
			
			# Initialize global variables
			if i == 0: self.init_totals(imdata, maskdata)

			def mask_deviant(threshold):
				delta = np.divide(np.abs(imdata - mean), mean)
				x,y,z = np.where(delta > threshold)
				return x,y,z

			binmask = ex.binmask()
			x,y,z = mask_deviant(opts.delta)
			binmask[x,y] = 0
			x,y = np.where( binmask == 0 )
			imdata[x,y,:] = 0
			if opts.verbose: imshow(imdata)
			self.sums  += imdata # masked imagedata
			self.maxes  = np.maximum(self.maxes, imdata)
			self.mins   = np.minimum(self.mins, imdata)

		# Save global variables
		if opts.do_save:
			np.save(affix('counts'), self.counts)
			np.save(affix('sums'), self.sums)
			np.save(affix('maxes'), self.maxes)
			np.save(affix('mins'), self.mins)
		return self.counts, self.sums, self.maxes, self.mins


	def compile(self):
		# Iterate through all data files
		sources = get_sources()
		for i, im_path in enumerate(sources):
			if opts.n and i == opts.n: break

			ex        = Example(im_path)
			imdata   = ex.imdata
			maskdata = ex.maskdata
			print('%6d/%d %s' % (i, len(sources), ex.bname))

			# Initialize global variables
			if i == 0: self.init_totals(imdata, maskdata)
			# Update global variables
			binmask = ex.binmask()
			self.counts += binmask
			# Mask current image
			xmask, ymask = np.where( binmask == 0 )
			imdata[xmask, ymask, :] = 0
			if opts.verbose: imshow(imdata)
			self.sums  += imdata # masked imagedata
			self.maxes  = np.maximum(self.maxes, imdata)
			self.mins   = np.minimum(self.mins, imdata)

		# Save global variables
		if opts.do_save:
			np.save(affix('counts'), self.counts)
			np.save(affix('sums'), self.sums)
			np.save(affix('maxes'), self.maxes)
			np.save(affix('mins'), self.mins)
		return self.counts, self.sums, self.maxes, self.mins

def affix(stem):
	if opts.load_recompiled_data:
		return '%s-n%d-d%f.npy' % (stem, opts.n, opts.delta)
	else:
		return '%s-n%d.npy' % (stem, opts.n)

def get_sources():
	return glob.glob(os.path.join(IMG_DIR, '*.jpg'))
	
def load():
	counts = np.load(affix('counts'))
	sums   = np.load(affix('sums'))
	maxes  = np.load(affix('maxes'))
	mins   = np.load(affix('mins'))
	return counts, sums, maxes, mins

def imsave(imdata, fpath):
	Image.fromarray(imdata.astype(np.uint8)).save(fpath)
def imshow(imdata):
	plt.imshow(imdata)
	plt.show()
def Imshow(imdata):
	Image.fromarray(imdata.astype(np.uint8)).show()

# Examine outputs manually
def show(counts, sums, maxes, mins):
	# Simple average
	# avg_img = np.divide( sums, np.expand_dims( counts, 2 ) ).astype(np.uint8)
	# ax = plt.subplot(121)
	# ax.imshow(avg_img)
	# # Max values for each pixel, leaving only completely masked holes
	# ax = plt.subplot(122)
	# ax.imshow(maxes)
	# # Mask
	# plt.show()
	Imshow(maxes)

def get_mask_categories():
	import csv
	with open('object150_info.csv') as csvfile:
		reader = csv.reader(csvfile)
		reader.__next__() # skip header line
		cats = [ row[5] for row in reader ]
	return cats

def show_masks():
	sources = get_sources()
	for i in range(opts.n or len(sources)):
		ex = Example(sources[i])
		imshow(ex.maskdata)
		imshow(ex.masked())
		imsave(ex.masked(), 'floor-mask.png')

def list_mask_cats():
	from collections import OrderedDict
	cats = get_mask_categories()
	dic = OrderedDict()
	sources = get_sources()
	for i in range(opts.n or len(sources)):
		ex  = Example(sources[i])
		catids = np.unique( ex.maskdata )
		for x in catids:
			dic[x] = cats[x]
	for k in dic:
		print( '%3d\t%s' % (k, dic[k] ))


if __name__ == '__main__':
	print(opts.act)
	if 'compile' in opts.act or len(opts.act) == 0:
		Compiler().compile()
	elif 'show' in opts.act:
		show(*load())
	elif 'show_masks' in opts.act:
		show_masks()
	elif 'recompile' in opts.act:
		outs = Compiler().recompile()
		plt.imshow(outs[2])
		plt.show()
	elif 'list' in opts.act:
		list_mask_cats()
	else:
		raise Exception('unknown action(s) %s' % str(opts.act))
