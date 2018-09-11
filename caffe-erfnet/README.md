# caffe-erfnet
**This is a modified version of [Caffe](https://github.com/BVLC/caffe),[Caffe-enet
](https://github.com/TimoSaemann/caffe-enet/tree/22d356c956cdc5e752e6d40612e4f6c60fc8f471/) and [PSPNet](https://github.com/hszhao/PSPNet/)

The interp layer in [PSPNet](https://github.com/hszhao/PSPNet/) was combined into [caffe-enet
](https://github.com/TimoSaemann/caffe-enet/tree/22d356c956cdc5e752e6d40612e4f6c60fc8f471/). And CuDNN5.0 in [caffe-enet
](https://github.com/TimoSaemann/caffe-enet/tree/22d356c956cdc5e752e6d40612e4f6c60fc8f471/) has been updated to CuDNN6.0. 

Credit: Alex Kendall for his [caffe-segnet](https://github.com/alexgkendall/caffe-segnet) repository,Timo Sämann for his [caffe-enet
](https://github.com/TimoSaemann/caffe-enet/tree/22d356c956cdc5e752e6d40612e4f6c60fc8f471/) repository, Hengshuang Zhao for his [PSPNet](https://github.com/hszhao/PSPNet/) repository,which are the bases for this. 

### Installation
For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe). To enable cuDNN for GPU acceleration, cuDNN v6.0 is needed. 
The code has been tested successfully on Ubuntu 16.04 with CUDA 8.0.

## How to use it
Please find a tutorial on how to use it in the [ENet](https://github.com/TimoSaemann/ENet) repository.

## License
This extension to the Caffe library is released under a creative commons license which allows for personal and research use only. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/

