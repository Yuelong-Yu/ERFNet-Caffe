#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script reads images from file, make predictions using pretrained models and write video
"""
# This is a modified version by 
__author__ = 'Yuelong Yu'
__university__ =  'Shanghai Jiao Tong University'
__email__ = 'yuyuelongfly@163.com'
__data__ = '11th September, 2018'

# The original version of the script is by Alex Kendall, https://github.com/alexgkendall/SegNet-Tutorial/tree/master/Scripts

import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

sys.path.append('/usr/local/lib/python2.7/site-packages')
caffe_root = '/ERFNet-Caffe/caffe-erfnet/'        # Change to your Caffe directory 
sys.path.insert(0, caffe_root + 'python')
import caffe

def main(args):
	net = caffe.Net(args.model,args.weights,caffe.TEST)
	caffe.set_mode_gpu()
	input_shape = net.blobs['data'].data.shape
	output_shape = net.blobs['Deconvolution23_deconv'].data.shape

	label_colours = cv2.imread(args.colours).astype(np.uint8)

	cv2.namedWindow("Input")
	cv2.namedWindow("ERFNet")

	#--------------------load image sequence from the file-----------------------------------------
	cap = cv2.VideoCapture("/media/yly/AIvehicle/CITYSCAPES_DATASET/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_02_copy/%04d.png") # Change to your directory that save the images, the images requires to be named in the formate 0000.png,0001.png,0002.png etc.

	#----------------------write results into video---------------------
	videoWriter=cv2.VideoWriter('/media/yly/AIvehicle/CITYSCAPES_DATASET/leftImg8bit_demoVideo/leftImg8bit/demoVideo/ERFNet.avi',cv2.VideoWriter_fourcc('X','V','I','D'),24,(1024,1024)) # change the frame rate and size of video

	rval = True

	while rval:
		start = time.time()
		rval, frame = cap.read()

		if rval == False:
			break

		end = time.time()
		print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

		start = time.time()
		frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
		input_image = frame.transpose((2,0,1))
		# input_image = input_image[(2,1,0),:,:] # May be required if you do not open your data with opencv
		input_image = np.asarray([input_image])
		end = time.time()
		print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

		start = time.time()
		out = net.forward(data=input_image)
		end = time.time()
		print '%30s' % 'Executed ERFNet in ', str((end - start)*1000), 'ms'

		start = time.time()
		#segmentation_ind = np.squeeze(net.blobs['argmax'].data)
		prediction = net.blobs['Deconvolution23_deconv'].data[0].argmax(axis=0)
		segmentation_ind = np.squeeze(prediction)


		segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
		segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
		segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

		#-----------Note: the color of the LUT in ENet is reversed--------------------
		label_colours_bgr = label_colours[..., ::-1] 
		
		cv2.LUT(segmentation_ind_3ch,label_colours_bgr,segmentation_rgb)
		#segmentation_rgb = segmentation_rgb.astype(float)/255
		
		segmentation_rgb = segmentation_rgb.astype(np.uint8)

		end = time.time()
		print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

		cv2.imshow("Input", frame)
		cv2.imshow("ERFNet", segmentation_rgb)

		key = cv2.waitKey(1)
		#----------write video----------------
		result=np.concatenate((frame,segmentation_rgb),axis=0)  # concatenate the input image and prediction vertically
		videoWriter.write(result)
		#videoWriter.write(segmentation_rgb)

		if key == 27: # exit on ESC
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/ERFNet-Caffe/prototxts/erfnet_deploy_mergebn.prototxt')
    parser.add_argument('--weights', type=str, default='/ERFNet-Caffe/weights/erfnet_cityscapes_mergebn.caffemodel')
    parser.add_argument('--colours', type=str, default='/ERFNet-Caffe/scripts/cityscapes19.png')
    args = parser.parse_args()
    main(args)
