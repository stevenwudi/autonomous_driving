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

The goal of this part is to classify images using some of the state of the art Deep Neural Networks (VGG, ResNet and InceptionV3). These architectures have been trained on different datasets (TT100k and KITTI). Belgium dataset has been also used.

**Implementation**

Firstly, the performance of a pretrained VGG model is evaluated for traffic sign classification with the database TT100k calculating the accuracy on train and test sets. Some changes in the configuration have been also done applying a crop and a resize to the images and an ImageNet and mean normalization. After that, the VGG network is also trained from scratch on another dataset, KITTI which consists in pedestrians, cyclists and vehciles images. 
The performance of two more models, ResNet and InceptionV3, has been also tested using the TT100k dataset from scratch and fine-tuning. 
Finally, the performance of our network has been boosted applying data augmentation with Horizontal Flip, Vertical Flip and zoom of 1.5, tuning the hyperparameters like learning rate and optimizer, adding layers in the classification block such as Batch Normalization, Dropout and Gaussian Noise.

**Level of completeness of the goals**

VGG:
- Training from scratch on TT100K dataset.
- Evaluating the techniques of cropping and resizing and pre-processing such as Imagenet normalization and mean normalization.
- Transfering the learning to the BelgiumTS dataset.
- Training from scratch and using fine-tuning on the KITTI dataset (comparing them).
- Boosting the performance of the network.

ResNet:
- Evaluating the model on TT100k comparing the training from scratch and using fine-tuning.

InceptionV3:
- Evaluating the model on TT100k comparing the training from scratch and using fine-tuning.


## Object Detection
**Introduction**

The goal of this part is to detect objects in the images using some of the state of the art Deep Neural Networks (YOLO, Tiny-YOLO and SSD) trained on different datasets TT110K, which consists in traffic signs images, and Udacity, which consists in cars, pedestrians and trucks images.

**Implementation**

During these weeks the performance of different networks as YOLO, Tiny-YOLO and SSD is evaluated for detection with the database TT100k and Udacity. The f-score and the frame per second (fps) is calculated separately in train, validation and test sets in order to analyze the results.
Finally, the performance of our network has been boosted applying data agumentation with Horizontal Flip and tuning 
hyperparameters like the optimizer (RMSprop and SGD) and the learning rate (from 0.00001 to 0.0001) following the configuration of the best results in the boosting part of Object Recognition.

**Level of completeness of the goals**

YOLO and Tiny-YOLO:
- Evaluating both networks on TT100K dataset.
- Analyzing the dataset
- Calculating the f-score and FPS on train, validation and test sets.
- Training the network on Udacity dataset.
- Boosting the performance of the both networks.

SSD: 
- Implementing the SSD network.
- Evaluating it on TT100K and Udacity.

## Image Semantic Segmentation
**Introduction**

The goal of this part is to segment each object of the scene giving its associated label using any state-of-the-art network.

**Implementation**

During theses weeks the performance of different models as FCN8 and SegNet has been evaluated using several datasets (CamVid, Cityscapes, Synthia and Polyps). To understand its behavior, the FCN8 and SegNet papers has been read and we also have done the corresponding summaries.
Finally, the performance of our network has been boosted applying data augmentation and making tuning of the metaparameters. 


**Level of completeness of the goals**

FCN:
- Reading the paper.
- Training and evaluating the FCN8 model on the CamVid dataset.
- Training/fine-tuning the model on the datasets Synthia, Cityscapes and Polyps.
- Boosting the performance of the network with Camvid dataset.

SegNet:
- Reading the paper.
- Implementing the model.
- Training the model on CamVid and Cityscapes datasets.

Tiramisu:
- Reading the paper.
- Implementing the model.
- Training the model on CamVid dataset.

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
- `ssd_detection.py` using both datasets (TT100K and Udacity)

Segmentation:
- `camvid_segmentation.py` using CamVid dataset (11 classes)
- `cityscapes_segmentation.py` using Cityscapes dataset (20 classes)
- `synthia_segmentation.py` using Synthia dataset (20 classes)
- `polyps_segmentation.py` using Polyps dataset (4 classes)

**Modules Used**

Object Recognition:
- VGG, ResNet and InceptionV3

Object Detection:
- YOLO, Tiny-YOLO and SSD

Segmentation:
- FCN, SegNet and Tiramisu

**F-Score and FPS**

To calculate the f-score and FPS on train, validation and test sets, go to the folder `VR-Team4/code` 
```
    python eval_detection_fscore.py MODEL_NAME DATASET_NAME WEIGHTS_PATH TEST_FOLDER_PATH
```
where the `MODEL_NAME` and the `DATASET_NAME` will be the name of the network and the dataset to be performed, respectively. For the `WEIGHTS_PATH` and the `TEST_FOLDER_PATH` will be the paths for the weights and the test folder with the images to be tested.

Other arguments (optionals) that will affect the final performance of the network will be ...
- `--detection-threshold NUM` Minimum confidence value for a prediction to be considered (between 0 and 1). By default = 0.5
- `--nms-threshold NUM` Non-Maxima Supression threshold (between 0 and 1). By default = 0.2
- `--display BOOLEAN` Display or not the image, the predicted bounding boxes and the ground truth bounding boxes. By default = FALSE
- `--ignore-class LIST_OF_INGNORED_CLASS` List of classes to be ignore from predictions. 

## References (Summary)
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/VGG%20Summary.md)**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/ResNet.md)**

J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. arXiv
preprint arXiv:1506.02640, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/YOLO.md)**

W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed, “SSD: Single shot multibox detector,” arXiv:1512.02325, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/SSD.md)**

J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3431–3440, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/FCN.md)**

V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” arXiv preprint arXiv:1511.00561, 2015. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/Segnet.md)**

S. Jégou, M. Drozdzal, D. Vázquez, A. Romero and Y. Bengio, “The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation“, arXiv preprint arXiv:1611.09326, 2016.
