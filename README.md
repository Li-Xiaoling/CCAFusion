
## Platform
Python 3.7  
Pytorch >=0.4.1  

The testing datasets are included in "images".

The results iamges are included in "outputs".

## Training Dataset

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) is utilized to train our auto-encoder network.

[VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/) is utilized to train the CCAF modules.


##  Application to Salient Object Detection

To demonstrate the effectiveness of the proposed fusion method on highlevel vision tasks, You can feed the source images and fused results into the salient object detection method [1], respectively.

[1] J.-J. Liu, Z.-A. Liu, P . Peng, and M.-M. Cheng, “Rethinking the ushape structure for salient object detection,” IEEE Trans. Image Process., vol. 30, pp. 9030–9042, 2021.


# Citation

@article{li2023ccafusion,
  title={CCAFusion: Cross-modal Coordinate Attention Network for Infrared and Visible Image Fusion},
  author={Li, Xiaoling and Li, Yanfeng and Chen, Houjin and Peng, Yahui and Pan, Pan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}


If you have any question about this code, feel free to reach me(x.l.li@bjtu.edu.cn) 



