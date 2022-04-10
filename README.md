# C-OF
Pytorch implementation for paper: A Real-Time and Long-Term Face Tracking Method Using Convolutional Neural Network and Optical Flow for Internet of Things

## Introduction

The development of Internet of Things (IoT) stimulates many research works related to Multimedia Communication Systems (MCS), such as human face detection and tracking. This trend drives numerous progressive methods. Among these methods, deep learning based method can spot face patch in an image effectively and accurately. Many people consider the face tracking as face detection, but they are two different techniques. Face detection focuses on single image, whose shortcoming is obvious, such as unstable and unsmooth face position when adopted on a sequence of continuous images; computing expensive due to its heavily relying on Convolutional Neural Networks (CNN) and limited detection performance on edge device. To overcome these defects, this paper proposes a novel face tracking strategy by combining CNN and Optical Flow, namely C-OF, which achieves an extremely fast, stable and long-term face tracking system. Two key things for commercial applications are the stability and smoothness of face positions in a sequence of image frames, which can provide more probability for face biological signal extracting, silent face anti-spoofing and facial expression analysis in the fields of IoT-based MCS. Our method captures face patterns in every two consequent frames via optical flow to get rid of the unstable and unsmooth problems. Moreover, an innovative metric for measuring the stability and smoothness of face motion is designed and adopted in our experiments. The experimental results illustrate that our proposed C-OF outperforms both face detection and object tracking methods.

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/imgs/C-OF-procedure.png" width=800/></div>

## Requirements

python==3.6.9

torch==1.4.0

torchvision==0.5.0

numpy==1.18.2

tqdm==4.45.0

facenet_pytorch

...

## How to use

 * You can download our testing data from Dropbox shared link by [here](https://www.dropbox.com/sh/fcks3k2l9xs36ze/AABlXm3FY3pMzStNrPktYKdRa?dl=0).
 * Please unzip the file to ./data/.
 * You can use your video or camera flames as inputs of our method. If you use local video, please put it in a proper place and amend the path in line 269 in cof_main.py, or use line 272-274 for camera flames.
 * Run cof_main.py.
 * C++ version using NCNN is available [here](https://github.com/HansRen1024/Face-Tracking-Using-CNN-and-Optical-Flow).

## Example

![image](https://github.com/HansRen1024/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_1.gif)

![image](https://github.com/HansRen1024/Face-Tracking-Based-on-OpenTLD-and-RNet/blob/master/example/saved_.gif)

## Performance

#### Stability

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/imgs/C-OF_stability.png" width=800/></div>

#### Smoothness

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/results/point_route/active_camera_center.png" width=800/></div>

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/results/point_route/active_human_center.png" width=800/></div>

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/results/point_route/active_illumination_center.png" width=800/></div>

<div align=center><img src="https://github.com/HansRen1024/C-OF/blob/main/results/point_route/static_human_center.png" width=800/></div>

## Citation

If you find this work helpful for your research, please cite the following paper:

    @article{ren2021real,
      title={A Real-Time and Long-Term Face Tracking Method Using Convolutional Neural Network and Optical Flow in IoT-Based Multimedia Communication Systems},
      author={Ren, Hanchi and Hu, Yi and Myint, San Hlaing and Hou, Kun and Zhang, Xiuyu and Zuo, Min and Zhang, Chi and Zhang, Qingchuan and Li, Haipeng},
      journal={Wireless Communications and Mobile Computing},
      volume={2021},
      year={2021},
      publisher={Hindawi}
    }

## References

1. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503. [PDF](https://arxiv.org/pdf/1604.02878.pdf)
2. Nam, H., & Han, B. (2016). Learning multi-domain convolutional neural networks for visual tracking. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4293-4302). [PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)

## Acknowledgement

We used pretrained model and relevant APIs from facenet-pytorch (https://github.com/timesler/facenet-pytorch). Thanks for their excellent work very much.
