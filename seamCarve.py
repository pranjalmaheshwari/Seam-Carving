#!/usr/bin/env
#author@Pranjal

import sys
import numpy as np
import logging
import cv2
import time
logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)


def seam(energy_img,helper_imgX,helper_imgY,ts):
	logging.debug('init time : '+str(time.time() - ts))
	ret_imgX = np.empty(shape=(helper_imgX.shape[0],helper_imgX.shape[1]-1),dtype=int)
	ret_imgY = np.empty(shape=(helper_imgY.shape[0],helper_imgY.shape[1]-1),dtype=int)
	dp = np.empty(shape = helper_imgX.shape,dtype=int)
	trck = np.empty(shape = helper_imgX.shape,dtype=int)
	
	logging.debug('dp init time : '+str(time.time() - ts))

	for c in range(helper_imgX.shape[1]):
		dp[0][c] = energy_img[helper_imgY[0][c]][helper_imgX[0][c]]
	
	logging.debug('dp start time : '+str(time.time() - ts))

	for r in range(1,helper_imgX.shape[0]):

		c = 0
		dp[r][c] = 9999999
		i = c
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		i += 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]

		xRange = helper_imgX.shape[1]-1
		for c in range(1,xRange):
			dp[r][c] = 9999999
			i = c - 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			i += 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			i += 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]
		
		c = xRange
		dp[r][c] = 9999999
		i = c - 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		i += 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]

	logging.debug('dp end time : '+str(time.time() - ts))

	min_index = np.argmin(dp[helper_imgX.shape[0]-1,:])

	for r in range(helper_imgX.shape[0]-1,-1,-1):
		index = 0
		for c in range(helper_imgX.shape[1]):
			if(c != min_index):
				ret_imgX[r][index] = helper_imgX[r][c]
				ret_imgY[r][index] = helper_imgY[r][c]
				index += 1
		min_index = trck[r][min_index]

	logging.debug('end time : '+str(time.time() - ts) + '\n\n')

	return ret_imgX,ret_imgY

def seamCarve(img_string,rows,cols,output_string='.jpg',a=0,b=0,c=255):
	ts = time.time()
	img = cv2.imread(img_string,1)
	if(rows > img.shape[0] and cols > img.shape[1]):
		print 'Invalid Input for rows and columns'
		exit()
	#energy_img = cv2.Laplacian(img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3,-1)
	g1 = cv2.pyrDown(img)
	g2 = cv2.pyrUp(g1)
	energy_img = cv2.cvtColor(img-g2,cv2.COLOR_BGR2GRAY)
	helper_imgX = np.empty(shape = energy_img.shape,dtype=int)
	helper_imgY = np.empty(shape = energy_img.shape,dtype=int)

	for r in range(energy_img.shape[0]):
		for c in range(energy_img.shape[1]):
			helper_imgX[r][c] = c
			helper_imgY[r][c] = r
	r1 = img.shape[0]
	c1 = img.shape[1]
	logging.debug('start time : ' + str(time.time() - ts) + '\n\n')
	logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
	while(r1 > rows or c1 > cols):
		if(c1 > cols):
			helper_imgX,helper_imgY = seam(energy_img,helper_imgX,helper_imgY,ts)
			c1 -= 1
			logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
		if(r1 > rows):
			energy_img = energy_img.T
			helper_imgX = helper_imgX.T
			helper_imgY = helper_imgY.T
			helper_imgY,helper_imgX = seam(energy_img,helper_imgY,helper_imgX,ts)
			energy_img = energy_img.T
			helper_imgX = helper_imgX.T
			helper_imgY = helper_imgY.T
			r1 -= 1
			logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
	used_mask = np.zeros(shape = img.shape,dtype='uint8')
	used_mask[:,:,0] = a*used_mask[:,:,0]
	used_mask[:,:,1] = b*used_mask[:,:,1]
	used_mask[:,:,2] = c*used_mask[:,:,2]
	output_img = np.zeros(shape = (rows,cols,3),dtype='uint8')
	for r in range(helper_imgX.shape[0]):
		for c in range(helper_imgX.shape[1]):
			used_mask[helper_imgY[r][c],helper_imgX[r][c],:] = img[helper_imgY[r][c],helper_imgX[r][c],:]
			output_img[r,c,:] = img[helper_imgY[r][c],helper_imgX[r][c],:]
	if(output_string != ''):
		cv2.imwrite('output_img'+output_string,output_img)
		cv2.imwrite('energy_img'+output_string,energy_img)
		cv2.imwrite('used_mask'+output_string,used_mask)

def helper_pyramid(disp_img,img,pyramidHeight):
	disp_img[-img.shape[0]:,:img.shape[1],:] = img
	if(pyramidHeight > 1):
		helper_pyramid(disp_img[:,img.shape[1]:,:],cv2.pyrDown(img),pyramidHeight-1)

def showGaussianPyramid(input_string,pyramidHeight=3,output_string='.jpg'):
	img = cv2.imread(input_string,1)
	disp_img = np.empty(shape = (img.shape[0],2*img.shape[1],img.shape[2]))
	helper_pyramid(disp_img,img,pyramidHeight)
	if(output_string != ''):
		cv2.imwrite('GaussianPyramid'+output_string,disp_img)

def getGaussianPyramid(input_string,pyramidHeight=3):
	img = cv2.imread(input_string,1)
	ret_imgs = [img]
	while(pyramidHeight > 1):
		img = cv2.pyrDown(img)
		ret_imgs.append(img)
		pyramidHeight -= 1
	return (ret_imgs)

def seamBenefit(energy_img,helper_imgX,helper_imgY,ts):
	logging.debug('init time : '+str(time.time() - ts))
	ret_imgX = np.empty(shape=(helper_imgX.shape[0],helper_imgX.shape[1]-1),dtype=int)
	ret_imgY = np.empty(shape=(helper_imgY.shape[0],helper_imgY.shape[1]-1),dtype=int)
	dp = np.empty(shape = helper_imgX.shape,dtype=int)
	trck = np.empty(shape = helper_imgX.shape,dtype=int)
	
	logging.debug('dp init time : '+str(time.time() - ts))

	for c in range(helper_imgX.shape[1]):
		dp[0][c] = energy_img[helper_imgY[0][c]][helper_imgX[0][c]]
	
	logging.debug('dp start time : '+str(time.time() - ts))

	for r in range(1,helper_imgX.shape[0]):

		c = 0
		dp[r][c] = 9999999
		i = c
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		i += 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]

		xRange = helper_imgX.shape[1]-1
		for c in range(1,xRange):
			dp[r][c] = 9999999
			i = c - 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			i += 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			i += 1
			if(dp[r-1][i] < dp[r][c]):
				dp[r][c] = dp[r-1][i]
				trck[r][c] = i
			dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]
		
		c = xRange
		dp[r][c] = 9999999
		i = c - 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		i += 1
		if(dp[r-1][i] < dp[r][c]):
			dp[r][c] = dp[r-1][i]
			trck[r][c] = i
		dp[r][c] += energy_img[helper_imgY[r][c]][helper_imgX[r][c]]

	logging.debug('dp end time : '+str(time.time() - ts))

	min_index = np.argmin(dp[helper_imgX.shape[0]-1,:])
	min_x = min_index
	max_x = min_index

	for r in range(helper_imgX.shape[0]-1,-1,-1):
		index = 0
		for c in range(helper_imgX.shape[1]):
			if(c != min_index):
				ret_imgX[r][index] = helper_imgX[r][c]
				ret_imgY[r][index] = helper_imgY[r][c]
				index += 1
		min_x = min(min_x,min_index)
		max_x = max(max_x,min_index)
		min_index = trck[r][min_index]

	logging.debug('end time : '+str(time.time() - ts) + '\n\n')

	return ret_imgX,ret_imgY,min_x,max_x

def seamCarveBenefit(pyramid,level,rows,cols,output_string='.jpg',a=0,b=0,c=255):
	ts = time.time()
	img = pyramid[0] #cv2.imread(img_string,1)
	if(img.shape[0]/int(2**level) < img.shape[0] - rows and img.shape[1]/int(2**level) < img.shape[1] - cols):
		print 'Invalid Input for rows and columns'
		exit()
	#energy_img = cv2.Laplacian(img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3,-1)
	#g1 = cv2.pyrDown(img)
	#g2 = cv2.pyrUp(g1)
	energy_img = cv2.cvtColor(img-cv2.pyrUp(pyramid[1])[:img.shape[0],:img.shape[1],:],cv2.COLOR_BGR2GRAY)
	helper_imgX = np.empty(shape = energy_img.shape,dtype=int)
	helper_imgY = np.empty(shape = energy_img.shape,dtype=int)

	for r in range(energy_img.shape[0]):
		for c in range(energy_img.shape[1]):
			helper_imgX[r][c] = c
			helper_imgY[r][c] = r

	img_dummy = pyramid[level]
	energy_img_dummy = cv2.cvtColor(img_dummy-cv2.pyrUp(pyramid[level+1])[:img_dummy.shape[0],:img_dummy.shape[1],:],cv2.COLOR_BGR2GRAY)
	helper_imgX_dummy = np.empty(shape = energy_img_dummy.shape,dtype=int)
	helper_imgY_dummy = np.empty(shape = energy_img_dummy.shape,dtype=int)

	for r in range(energy_img_dummy.shape[0]):
		for c in range(energy_img_dummy.shape[1]):
			helper_imgX_dummy[r][c] = c
			helper_imgY_dummy[r][c] = r

	r1 = img.shape[0]
	c1 = img.shape[1]
	logging.debug('start time : ' + str(time.time() - ts) + '\n\n')
	logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
	while(r1 > rows or c1 > cols):
		if(c1 > cols):
			helper_imgX_dummy,helper_imgY_dummy,min_x,max_x = seamBenefit(energy_img_dummy,helper_imgX_dummy,helper_imgY_dummy,ts)
			min_x = min_x*(2**level)
			max_x = (max_x*(2**level)) + 1
			helper_imgX[:,min_x:max_x-1],helper_imgY[:,min_x:max_x-1] = seam(energy_img,helper_imgX[:,min_x:max_x],helper_imgY[:,min_x:max_x],ts)
			helper_imgX = np.delete(helper_imgX,max_x-1,1)
			helper_imgY = np.delete(helper_imgY,max_x-1,1)
			c1 -= 1
			logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
		if(r1 > rows):
			energy_img = energy_img.T
			helper_imgX = helper_imgX.T
			helper_imgY = helper_imgY.T
			energy_img_dummy = energy_img_dummy.T
			helper_imgX_dummy = helper_imgX_dummy.T
			helper_imgY_dummy = helper_imgY_dummy.T
			helper_imgY_dummy,helper_imgX_dummy,min_x,max_x = seamBenefit(energy_img_dummy,helper_imgY_dummy,helper_imgX_dummy,ts)
			min_x = min_x*(2**level)
			max_x = (max_x*(2**level)) + 1
			helper_imgY[:,min_x:max_x-1],helper_imgX[:,min_x:max_x-1] = seam(energy_img,helper_imgY[:,min_x:max_x],helper_imgX[:,min_x:max_x],ts)
			helper_imgX = np.delete(helper_imgX,max_x-1,1)
			helper_imgY = np.delete(helper_imgY,max_x-1,1)
			energy_img_dummy = energy_img_dummy.T
			helper_imgX_dummy = helper_imgX_dummy.T
			helper_imgY_dummy = helper_imgY_dummy.T
			energy_img = energy_img.T
			helper_imgX = helper_imgX.T
			helper_imgY = helper_imgY.T
			r1 -= 1
			logging.debug('rows : ' + str(r1) + '  cols : ' + str(c1) + '  time : ' + str(time.time() - ts) + '\n\n')
	used_mask = np.zeros(shape = img.shape,dtype='uint8')
	used_mask[:,:,0] = a*used_mask[:,:,0]
	used_mask[:,:,1] = b*used_mask[:,:,1]
	used_mask[:,:,2] = c*used_mask[:,:,2]
	output_img = np.zeros(shape = (rows,cols,3),dtype='uint8')
	for r in range(helper_imgX.shape[0]):
		for c in range(helper_imgX.shape[1]):
			used_mask[helper_imgY[r][c],helper_imgX[r][c],:] = img[helper_imgY[r][c],helper_imgX[r][c],:]
			output_img[r,c,:] = img[helper_imgY[r][c],helper_imgX[r][c],:]
	if(output_string != ''):
		cv2.imwrite('output_img'+output_string,output_img)
		cv2.imwrite('energy_img'+output_string,energy_img)
		cv2.imwrite('used_mask'+output_string,used_mask)

def benefit(img_string,rows,cols,level=1,output_string='.jpg'):
	pyramid = getGaussianPyramid(img_string,level+2)
	seamCarveBenefit(pyramid,level,rows,cols,output_string,0,0,255)



