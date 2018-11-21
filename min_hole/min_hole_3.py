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
assert os.path.exists(IMG_DIR), IMG_DIR
assert os.path.exists(MASK_DIR), MASK_DIR

# Nix:
# 12 person;individual;someone;somebody;mortal;soul
# 19 chair
# 31 seat
# 74 computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system
# 75 swivel;chair
# 98 bottle
# 141 crt;screen
# 143 monitor;monitoring;device
# 147 glass;drinking;glass
MASKED_CATEGORIES = [12, 19, 31, 74, 75, 98, 147, 141, 143]
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
		binmask = np.ones_like( self.maskdata, np.uint8 )
		for category in MASKED_CATEGORIES:
			xmask, ymask = np.where( self.maskdata == category )
			binmask[xmask, ymask] -= 1
		if opts.spread:
			for offset in range(opts.spread):
				if offset:
					binmask[offset:,:] *= binmask[:-offset,:] # down
					binmask[:,offset:] *= binmask[:,:-offset] # right
					binmask[:-offset,:] *= binmask[offset:,:] # up
					binmask[:,:-offset] *= binmask[:,offset:] # left
		assert binmask.min() == 0, binmask.min()
		return np.expand_dims( binmask, 2 )

	def mask(self, binmask):
		imdata  = self.imdata.copy()
		imdata *= binmask
		return imdata

	# Return imdata with masked regions set to 0
	def masked(self):
		return self.mask(self.binmask())

	# Return imdata with masked regions set to 255
	def minmasked(self):
		imdata  = self.imdata.copy()
		binmask = self.binmask()
		maxvals = 255 * (1-binmask)
		return self.mask(binmask) + maxvals

class Compiler(object):
	def init_totals(self, imdata, maskdata):
		self.counts = np.expand_dims( np.zeros_like( maskdata, np.uint32 ), 2 )
		self.sums   = np.zeros_like( imdata,   np.uint32 )
		self.maxes  = np.zeros_like( imdata,   np.uint8 )
		self.immaxes  = np.zeros_like( imdata,   np.uint8 )
		self.immins   = np.ones_like(  imdata,   np.uint8 ) * 255
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
			print('%6d/%d %s' % (i, len(sources), ex.bname))

			# Initialize global variables
			if i == 0: self.init_totals(ex.imdata, ex.maskdata)
			# Update global variables
			self.counts += ex.binmask()
			# Mask current image
			masked = ex.masked()
			minmasked = ex.minmasked()
			if opts.verbose: imshow(masked)
			self.sums  += masked # masked imagedata
			self.maxes  = np.maximum(self.maxes, masked)
			self.immaxes  = np.maximum(self.immaxes, ex.imdata)
			self.mins   = np.minimum(self.mins, minmasked)
			self.immins   = np.minimum(self.immins, ex.imdata)

			if i % 10 == 0:
				imsave(self.maxes, '%d.jpg' % i)
				imsave(self.mins, 'min-%d.jpg' % i)
				avg = np.divide(self.sums, self.counts)
				imsave(avg, 'avg-%d.jpg' % i)
				imsave(self.immaxes, 'immax-%d.jpg' % i)
				imsave(self.immins, 'immin-%d.jpg' % i)
				np.save(outpath('mask-%d.jpg' % i), ex.binmask())


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
	if opts.frames:
		print(opts.frames)
		return [os.path.join(IMG_DIR, '%05d.jpg' % int(f,10)) for f in opts.frames]
	else:
		return glob.glob(os.path.join(IMG_DIR, '*.jpg'))
	
def load():
	counts = np.load(affix('counts'))
	sums   = np.load(affix('sums'))
	maxes  = np.load(affix('maxes'))
	mins   = np.load(affix('mins'))
	return counts, sums, maxes, mins

def outpath(name_or_path):
	if opts.out and name_or_path == os.path.basename(name_or_path):
		return os.path.join(opts.out, name_or_path)
	return name_or_path

def imout(imdata, fpath):
	if opts.no_screen:
		imsave(imdata, fpath)
		print('saved to %s' % fpath)
	else:
		imshow(fpath)
def imsave(imdata, fpath):
	if opts.out and fpath == os.path.basename(fpath):
		fpath = os.path.join(opts.out, fpath)
		if not os.path.exists(opts.out): os.makedirs(opts.out)
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
		# imout(ex.maskdata, 'maskdat-%s' % ex.bname)
		imout(ex.masked(), 'masked-%s' % ex.bname)

def list_mask_cats():
	from collections import OrderedDict
	cats = get_mask_categories()
	dic = OrderedDict()
	sources = get_sources()
	catids = set()
	for i in range(opts.n or len(sources)):
		print(i)
		ex  = Example(sources[i])
		catids.update(set(np.unique( ex.maskdata )))
	for k in catids:
		print( '%3d\t%s' % (k, cats[k] ))


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
