# Visual Recognition for Autonomous Driving
The motivation of this project is to learn and develop a fully capable algorithm that will not only detect the different objects/vehicles in a road, but it will be able to recognise them and segment the image. One application for this project would be its implementation in an antonomous self-driving car. The project will be composed in 5 deliveries (one for each week).

## Authors
Team name: **Deep4Learning**
- _A. Casadevall, arnau.casadevall.saiz@gmail.com - [acasadevall](https://github.com/acasadevall)_
- _S. Castro, sergio.castro.latorre@gmail.com - [SCLSCL](https://github.com/SCLSCL)_
- _T. Pérez, tonipereztorres@gmail.com - [tpereztorres](https://github.com/tpereztorres)_
- _A. Salvador, adrisalva122@gmail.com - [AdrianSalvador](https://github.com/AdrianSalvador)_

## Report
The report can be found in the following Overleaf project: [Visual Recognition for Autonomous Driving](https://www.overleaf.com/read/jttxgmksgkjm)

## Presentation
The project presentation can be found in this [link](https://docs.google.com/presentation/d/1pATMrlv-86Eotm-Z1qS7ohpkkS3RFFPdB2qlYrZ8-4Y).

## Weights
The weights of the different models can be found [here](https://drive.google.com/open?id=0B3z5gWH7cHJiWm1pUVRoOFd3dTQ).

## Object Recognition
**Introduction**

The goal of this part is to classify images using some of the state of the art deep neural networks (VGG, ResNet and InceptionV3). These architectures have been trained on different datasets (TT100k and KITTI). Belgium dataset has been also used.

**Implementation**

Firstly, the performance of a pretrained VGG model is evaluated for traffic sign classification with the database TT100k calculating the accuracy on train and test sets. Some changes in the configuration have been also done applying a crop and a resize to the images and an ImageNet and mean normalization. After that, the VGG network is also trained from scratch on another dataset, KITTI which consists in pedestrians, cyclists and vehciles images. 
The performance of two more models, ResNet and InceptionV3, has been also tested using the TT100k dataset from scratch and fine-tuning. 
Finally, the performance of our network has been boosted applying data augmentation with Horizontal Flip, Vertical Flip and zoom of 1.5, tuning the hyperparameters like learning rate and optimizer, adding layers in the classification block such as Batch Normalization, Dropout and Gaussian Noise.

All the results can be followed in our [Overleaf project](https://www.overleaf.com/read/wwstzqxkjcxb) or in [these slides](https://docs.google.com/presentation/d/1pATMrlv-86Eotm-Z1qS7ohpkkS3RFFPdB2qlYrZ8-4Y).

## Object Detection
**Introduction**

**Implementation**


## How to Execute the Code
Go to the folder `VR-Team4/code` 
```
    python train.py -c config/CONF_FILE.py -e EXT_NAME
```
where the `CONF_FILE.py` will be one of the _configurations file_ located in `code/config` and the `EXT_FOLDER` the name of the experimental folder where all the results will be saved.

**Configurations File**

Object Recognition:
- `tt100k_classif.py` using TT100K traffic signs dataset
- `belgium_classif.py` using Belgium traffic signs dataset
- `kitti_classif.py` using KITTI dataset

Object Detection:
- `tt100k_detection.py` using TT100K traffic signs dataset
- `udacity_detection.py` using Udacity dataset
- `ssd_detection.py` (Implementing...)

Segmentation (not implemented yet):
- [···]

**Modules Used**

Object Recognition:
- VGG, ResNet, InceptionV3

Object Detection:
- YOLO, Tiny-YOLO, SSD

Segmentation:

To calculate the f-score and FPS on train, validation and test datsets, go to the folder `VR-Team4/code` 
```
    python eval_detection_fscore.py -c config/CONF_FILE.py -e EXT_NAME -s DATASET_FOLDER
```
where the `CONF_FILE.py` will be one of the _configurations file_ located in `code/config`, the `EXT_FOLDER` the name of the experimental folder where all the results will be saved and the `DATASET_FOLDER` the path of the specified set (train, validation or test).


## References (Summary)
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/VGG%20Summary.md)**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/ResNet.md)**

J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. arXiv
preprint arXiv:1506.02640, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/YOLO.md)**

W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed, “SSD: Single shot multibox detector,” arXiv:1512.02325, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/SSD.md)**

