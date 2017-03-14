# Visual Recognition for Autonomous Driving
The motivation of this project is to learn and develop a fully capable algorithm that will not only detect the different objects/vehicles in a road, but it will be able to recognise them and segment the image. One application for this project would be its implementation in an antonomous self-driving car. The project will be composed in 5 deliveries (one for each week).

## Authors
Team name: **Deep4Learning**
- _A. Casadevall, arnau.casadevall.saiz@gmail.com - [acasadevall](https://github.com/acasadevall)_
- _S. Castro, sergio.castro.latorre@gmail.com - [SCLSCL](https://github.com/SCLSCL)_
- _T. Pérez, tonipereztorres@gmail.com - [tpereztorres](https://github.com/tpereztorres)_
- _A. Salvador, adrisalva122@gmail.com - [AdrianSalvador](https://github.com/AdrianSalvador)_

## Report
The report can be found in the following Overleaf project: [Visual Recognition for Autonomous Driving](https://www.overleaf.com/read/wwstzqxkjcxb)

## Presentation
The project presentation can be found in this [link] (https://docs.google.com/presentation/d/1pATMrlv-86Eotm-Z1qS7ohpkkS3RFFPdB2qlYrZ8-4Y/edit?usp=sharing).

## Weights

The weights of the different models can be found [here] (https://drive.google.com/open?id=0B3z5gWH7cHJiWm1pUVRoOFd3dTQ).

## Object Recognition
**Introduction**

The goal of this part is to classify images using some of the state of the art deep neural networks (VGG, ResNet and InceptionV3). These architectures have been trained on different datasets (TT100k and KITTI). Belgium dataset has been also used.

**Implementation**

Firstly, the performance of a pretrained VGG model is evaluated for traffic sign classification with the database TT100k calculating the accuracy on train and test sets. Some changes in the configuration have been also done applying a crop and a resize to the images and an ImageNet and mean normalization. After that, the VGG network is also trained from scratch on another dataset, KITTI which consists of pedestrians, cyclists and vehciles images. 
The performance of two more models, ResNet and InceptionV3, has been also tested using the TT100k dataset from scratch and fine-tuning. 
Finally, the performance of our network has been boosted applying data augmentation with Horizontal Flip, Vertical Flip and zoom of 1.5, tuning the hyperparameters like learning rate and optimizer, adding layers in the classification block such as Batch Normalization, Dropout and Gaussian Noise.

## Weekly Project Slides
- Google slides for [Week 1](https://docs.google.com/presentation/d/1A6hgbNn8N-Iq8MhSa_RPIyf87DBL6PCtoDzy1zqS5Xs/edit?usp=sharing)
- Google slides for [Week 2](https://docs.google.com/presentation/d/1Q69lmzPzgtc4lDw8dr9yyFY_T9JXhjJgL4ShyxFJk3M/edit?usp=sharing)
- Google slides for [Week 3](https://docs.google.com/presentation/d/1WuzVeTwUL65Dnki3vsBJgHXwKffrFanwXbmR_URkLQQ/edit?usp=sharing)
- Google slides for Week 4 (T.B.A.)
- Google slides for Week 5 (T.B.A.)
- Google slides for Week 6 (T.B.A.)
- Google slides for Week 7 (T.B.A.)

## References (Summary)
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/VGG%20Summary.md)**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770-778). **[Paper summary](https://github.com/acasadevall/VR-Team4/blob/master/Summaries/ResNet.md)**