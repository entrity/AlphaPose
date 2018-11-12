import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Constants for specifying location of data files
BASE_DIR = os.path.expanduser('~/mnt')
IMG_DIR  = os.path.join(BASE_DIR, 'first_third_person/pose_data/training_data/training/training')
MASK_DIR = os.path.join(BASE_DIR, 'first_third_person/Shift-Net_pytorch/datasets/Paris_StreetView_Dataset/Mask2')

counts = None # for each pixel (and channel): how many images have this pixel UNmasked
sums   = None # the sum value of all unmasked pixels for each location and channel
maxes  = None # the max value of all unmasked pixels for each location and channel

# Iterate through all data files
sources = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
for i, im_path in enumerate(sources):

	# Load image data and mask data
	bname = os.path.basename(im_path)
	print('%6d/%d %s' % (i, len(sources), bname))
	name, ext = os.path.splitext(bname)
	mask_path = os.path.join(MASK_DIR, '%s.png' % name)
	img  = Image.open(im_path)
	mask = Image.open(mask_path)

	# Build a masked image
	maskarray = np.array(mask.getdata()).reshape( mask.height, mask.width )
	xmask, ymask = np.where(maskarray == 12)
	imarray = np.array(img.getdata()).reshape(img.height, img.width, 3)
	imarray[xmask, ymask, :] = 0
	if os.environ.get('DBGPY'):
		Image.fromarray(imarray.astype(np.uint8)).save('%d.jpg' % i)	

	# Initialize global variables
	if counts is None:
		counts = np.ones_like( maskarray ) * len(sources)
		sums   = np.zeros_like( imarray )
		maxes  = np.zeros_like( imarray )

	# Update global variables
	counts[xmask, ymask] -= 1
	sums  += imarray
	maxes  = np.maximum(maxes, imarray)

# Save global variables
np.save('counts.npy', counts)
np.save('sums.npy', sums)
np.save('maxes.npy', maxes)

# Examine outputs manually
avg_img = np.divide( sums, np.expand_dims( counts, 2 ) )
ax = plt.subplot(121)
ax.imshow(avg_img)
ax = plt.subplot(122)
ax.imshow(maxes)
plt.show()
