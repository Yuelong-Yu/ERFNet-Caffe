#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script export the semantic segmentation of ERFNet for all test images in a file so that the IoU can be evaluated.
"""
# The original version of the script is by 
__author__ = 'Timo Sämann' # ENet
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'

# This is a modified version by 
__author__ = 'Yuelong Yu'
__university__ =  'Shanghai Jiao Tong University'
__email__ = 'yuyuelongfly@163.com'
__data__ = '11th September, 2018'

import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '/home/yly/ENet/caffe-enet/'   # Change to your Caffe directory 
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import time

EXTENSIONS = ['.jpg', '.png']

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def make_parser():
    parser = argparse.ArgumentParser()
    '''  
    ##-----------------encoder: if you trained encoder and decoder separately, uncomment this to visualize the results of encoder ---------------
    parser.add_argument('--model', type=str, default='/home/yly/ERFNet/prototxts/erfnet_encoder_deploy.prototxt')
    parser.add_argument('--weights', type=str, default='/home/yly/ERFNet/weights/snapshots_encoder/erfnet_encoder_cityscapes.caffemodel')
    '''
    ##-----------------decoder:Change the following directories----------------
    parser.add_argument('--model', type=str, default='/home/yly/ERFNet/prototxts/erfnet_deploy_mergebn.prototxt')
    parser.add_argument('--weights', type=str, default='/home/yly/ERFNet/weights/erfnet_cityscapes_mergebn.caffemodel')

    parser.add_argument('--colours', type=str, default='/home/yly/ERFNet/scripts/cityscapes19.png')
    parser.add_argument('--imagedir', type=str, default='/media/yly/AIvehicle/CITYSCAPES_DATASET/leftImg8bit/val/munster/') # Change to your path of test images 
    parser.add_argument('--out_dir', type=str, default='/media/yly/AIvehicle/prediction/gtFine/val/')
    parser.add_argument('--gpu', type=int, default=0, help='0: gpu mode active, else gpu mode inactive')

    return parser

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.model, args.weights, caffe.TEST)
    
    input_shape = net.blobs['data'].data.shape
    
    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.imagedir)) for f in fn if is_image(f)]

    for filename in filenames:

        input_image = cv2.imread(filename, 1).astype(np.float32)

        input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        
        input_image = np.asarray([input_image])

        start_time=time.time()
        #out = net.forward_all(**{net.inputs[0]: input_image})  # Run net forward in batches
        out = net.forward(data=input_image)
        fwt=time.time()-start_time
        print fwt
        ''' 
        ##-----------------encoder: Change the name of output blobs according to your prototxts---------------
        output_shape = net.blobs['Output_conv'].data.shape
        prediction = net.blobs['Output_conv'].data[0].argmax(axis=0)
        '''
        ##----------------decoder: Change the name of output blobs according to your prototxts-----------------
        output_shape = net.blobs['Deconvolution23_deconv'].data.shape
        prediction = net.blobs['Deconvolution23_deconv'].data[0].argmax(axis=0)
  
        prediction = np.squeeze(prediction)
        prediction = np.resize(prediction, (3, output_shape[2], output_shape[3]))
        prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

        prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
        label_colours_bgr = label_colours[..., ::-1]
        cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

        #cv2.imshow("ENet", prediction_rgb)
        #key = cv2.waitKey(0)

        if args.out_dir is not None:
            input_path_ext = filename.split(".")[-1]
            input_image_name = filename.split("/")[-1:][0].replace('.' + input_path_ext, '')
            #out_path_im = args.out_dir + input_image_name + '_erfnet' + '.' + input_path_ext
            out_path_gt = args.out_dir + input_image_name + '_erfnet_gt' + '.' + input_path_ext

            #cv2.imwrite(out_path_im, prediction_rgb) # color images for visualization
            cv2.imwrite(out_path_gt, prediction) #  label images, every pixel has an ID that represents the class






