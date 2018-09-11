# ERFNet-Caffe-version
Implementation of the Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation in caffe
<div align=center><img src="https://github.com/Yuelong-Yu/ERFNet-Caffe/blob/master/example_image/munich_000000_000019_leftImg8bit.png" width="512" height="256" alt="Test image from Cityscapes dataset"/><br>
Test image from Cityscapes dataset</div><br>

<div align=center><img src="https://github.com/Yuelong-Yu/ERFNet-Caffe/blob/master/example_image/munich_000000_000019_leftImg8bit_erfnet.png" width="512" height="256" alt="Semantic Segmentation of ERFNet"/><br>
Semantic Segmentation of ERFNet</div>

## Publications<br>
The deep neural network architecture is based on the following publication:<br>
"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. <br>

Several modifications were made:
- Instead of training encoder and decoder stage seperately in the above paper, the whole architecture is trained directly with an auxiliary loss after the encoder part to control the loss of encoder during training phase;<br>
- In the decoder part, the kernel size of deconvolution is 2x2 rather than 3x3 in the paper, since the 3x3 kernel will lead to odd size of feature map.And two Non-bt-1D are added in the decoder part to complement the shrinked kernel size.

## Visualize the prediction with python<br>
Firstly, change caffe_root in ERFNet-Caffe/scripts/test_segmentation.py to the absolute path of caffe; the original caffe version [BVLC/caffe](https://github.com/BVLC/caffe/) is enough for prediction.<br>
After that, you can visualize the prediction of ERFNet by running:
```
$ python test_segmentation.py 	--model ERFNet-Caffe/prototxts/erfnet_deploy_mergebn.prototxt \
				--weights ERFNet-Caffe/weights/erfnet_cityscapes_mergebn.caffemodel\
				--colours ERFNet-Caffe/scripts/cityscapes19.png \
				--input_image ERFNet-Caffe/example_image/munich_000000_000019_leftImg8bit.png \
				--out_dir ERFNet-Caffe/example_image/ 
```

## Training ERFNet<br>
- Compile ERFNet-Caffe/caffe-erfnet for training. Caffe-erfnet combines the interp layer in [PSPNet](https://github.com/hszhao/PSPNet/) and DenseImageData layer in [caffe-enet
](https://github.com/TimoSaemann/caffe-enet/tree/22d356c956cdc5e752e6d40612e4f6c60fc8f471/) to create auxiliary loss and data interface, respectively.<br>

- Change your net directory and snapshot_prefix directory in ERFNet-Caffe/prototxts/erfnet_solver.prototxt;<br>
- Change your source directory in ERFNet-Caffe/prototxts/erfnet_train_val.prototxt;<br>
- Change your directory of cityscapes data (images and labels) in ERFNet-Caffe/dataset/train_fine_cityscapes.txt and ERFNet-Caffe/dataset/eval_fine_cityscapes.txt. <br>

- Start the training from scratch:<br>
```
$ ERFNet-Caffe/caffe-erfnet/build/tools/caffe train -solver /ERFNet-Caffe/prototxts/erfnet_solver.prototxt
```
or start the training with the pretrained model:<br>
```
$ ERFNet-Caffe/caffe-erfnet/build/tools/caffe train -solver /ERFNet-Caffe/prototxts/erfnet_solver.prototxt -snapshot /ERFNet-Caffe/weights/erfnet_cityscapes.caffemodel
```

## Accelerate prediction (optional)<br>
Merge BatchNorm & Scale layers into Convolution layers; and remove dropout layer in test phase to accelerate prediction
```
$ python merge_bn_scale_droupout.py 	--model ERFNet-Caffe/prototxts/erfnet_deploy.prototxt \
				--weights ERFNet-Caffe/weights/erfnet_cityscapes.caffemodel\
				--output_model ERFNet-Caffe/prototxts/erfnet_deploy_mergebn.prototxt \
				--output_weights ERFNet-Caffe/weights/erfnet_cityscapes_mergebn.caffemodel
```
