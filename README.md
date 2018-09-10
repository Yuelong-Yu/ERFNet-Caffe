# ERFNet-Caffe-version
Implementation of the ERFNet for Real-Time Semantic Segmentation in caffe

## Publications<br>
The deep neural network architecture is based on the following publication:<br>
"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. <br>
Several modifications were made:<br>
1. Instead of training encoder and decoder stage seperately in the above paper, the whole architecture is trained directly with an auxiliary loss after the encoder part to control the loss of encoder during training phase;<br>
2. In the decoder part, the kernel size of deconvolution is 2x2 rather than 3x3 in the paper, since the 3x3 kernel will lead to odd size of feature map.And two Non-bt-1D are added in the decoder part to complement the shrinked kernel size.
