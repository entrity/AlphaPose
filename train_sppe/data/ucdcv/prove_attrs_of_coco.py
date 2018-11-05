import h5py

if __name__ == '__main__':
	data = h5py.File('../coco/annot_coco.h5', 'r')
	# Test bounding boxes are x1,y1,x2,y2 instead of x,y,w,h
	bbs = data['bndbox'].value
	n = bbs.shape[0]
	x = (bbs[:,:,0] < bbs[:,:,2]).sum()
	print('x1 < x2 %s\t %d %d' % (x == n, x, n))
	y = (bbs[:,:,1] < bbs[:,:,3]).sum()
	print('y1 < y2 %s\t %d %d' % (y == n, y, n))
	# Test whether keypoints are all within the bounding box
	kps = data['part'].value
	n = kps.shape[0]
	for i in range(n):
		body = kps[i,:,:]
		bb   = bbs[i,:,:].reshape(-1)
		for row in range(body.shape[0]):
			x,y = body[row,:]
			if x == 0 and y == 0: continue
			assert x >= bb[0], '(%d) kypt (%d %d) bb (%d %d %d %d)' % (i, x, y, *bb)
			assert x <= bb[2], '(%d) kypt (%d %d) bb (%d %d %d %d)' % (i, x, y, *bb)
			assert y >= bb[1], '(%d) kypt (%d %d) bb (%d %d %d %d)' % (i, x, y, *bb)
			assert y <= bb[3], '(%d) kypt (%d %d) bb (%d %d %d %d)' % (i, x, y, *bb)
