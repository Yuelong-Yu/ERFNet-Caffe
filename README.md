# ERFNet-Caffe-version
Implementation of the Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation in caffe

## Publications<br>
The deep neural network architecture is based on the following publication:<br>
"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. <br>
Several modifications were made:<br>
1. Instead of training encoder and decoder stage seperately in the above paper, the whole architecture is trained directly with an auxiliary loss after the encoder part to control the loss of encoder during training phase;<br>
2. In the decoder part, the kernel size of deconvolution is 2x2 rather than 3x3 in the paper, since the 3x3 kernel will lead to odd size of feature map.And two Non-bt-1D are added in the decoder part to complement the shrinked kernel size.

## Visualize the prediction with python<br>
Firstly, change caffe_root in ERFNet-Caffe/scripts/test_segmentation.py to the absolute path of caffe; the original caffe version [BVLC/caffe](https://github.com/BVLC/caffe/) is enough for prediction.<br>
After that, you can visualize the prediction of ERFNet by running:
```
$ python test_segmentation.py 	--model ERFNet/prototxts/erfnet_deploy_mergebn.prototxt \
				--weights ERFNet/weights/erfnet_cityscapes_mergebn.caffemodel\
				--colours ERFNet/scripts/cityscapes19.png \
				--input_image ERFNet/example_image/munich_000000_000019_leftImg8bit.png \
				--out_dir ERFNet/example_image/ 
```
