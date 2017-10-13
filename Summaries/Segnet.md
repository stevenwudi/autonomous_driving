# Segnet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

## Introduction & State of the art
Semantic Segmentation has a wide range of applications, from scene understanding to autonomous driving. With the deep learning techniques have accomplished big performances in handwritten recognition, speech and object detection. The motivation under Segnet is the need of mapping the low resolution features to input resolution for pixel-wise classification. 

Segnet is primarly motivated by road scene understanding applications where there is a necessity of modelling the appearance (road, etc), shape (car, pedestrian) and understand the spatial-relationship between different classes. It is important to delineate objects based on its shape and keep the boundary information. 

Segnet is topollogically identical to VGG16 but without the fully connected layers, making Segnet smaller and easier to train. Its key correspond to the decoder network that consists in hiercarchy of decoders one corresponding to each encoder. The appropiates decoders use max-pooling indices received from the input feature maps, which improves the boundary delineation, reduces the number of parameters and it allows the incorporation to any encoder-decoder architecture. 

Before deep learning techniques, the best performances among semantic pixel-wised relied on hand engineered features classifying pixels independently. More recent approaches predict the labels for all the pixels in a patch instead of only the centered pixel, giving better results and performances. 

New architectures particularly designed for segmentation have advanced the state-of-the-art by learning how to decode and or map low resolution representations in all of these architectures in the VGG16 classification network. 

## Architecture
Segnet has an encoder network and a corresponding decoder network with a final pixelwise classification layer. The encoder network is formed by 13 convolutional layers (same as VGG16), therefore, it can use the weigths from pre-trained classification applications for training. The fully connected layers are discarded in order to retain higher resolution maps. 

Each enconder performs a convolution with a filter bank to produce a set of features maps. Then the ReLu is applied with a maxpooling 2x2 window with stride 2 and the resulting output is sub-sampled by 2. Max-Pooling is used to achieve translation invariance over small spatial shifts. The increasingly loss image representation is not beneficial for segmentation as it blurrs boundary delineation. That is why the boundary information is capture and stored in the encoder feature maps before sub-sampling. In Segnet, there is an improving inside the boundary information caputre as it stores only the max-pooling indices (locations of the maximum feature value in each pooling window).

The decoders decode the feature maps using the memorized max-pooling indices from the corresponding feature map. These feature maps are then convolved with filter bank to produce dense feature maps with and applied batch normalization. The output of the softmax is a K channel image of probabilities where K is the number of classes. The predicted segmentation correspond to the class with maximum probability at each pixel.

## Decoder Variants
In order to compare Segnet with FCN, there has been implemented a reduced Segnet architecture named Segnet-Basic, with 4 encoders and 4 decoders. All the encoders perform max-pooling and sub-sampling and the decoders perform upsample with the received max-pooling indices. Batch Normalization and ReLu are also implemented as explaind before. Also, a 7x7 kernel is also used in order to provide a wide context for smooth labelling. 
FNC-Basic is also created which shares the characteristics of Segnet-Basic but with the decoding procedure of FCN. 

In Segnet, there is no learning involved in the upsampling process. Each decoder filter has the same number of channels as the upsampled feature maps. 
FCN is different as it has a dimensionally reduction effect in the encoded feature maps. The compressed K channels (K is the number of classes) final encoder layer are the input of the decoder network. The upsampling in this network is performed by inverse convolution using a fixed kernel of 8x8 (also named convolution).

## Training & Analysis
The dataset used is CamVid, which consists in 367 training and 233 testing RGB images at 360 x 480 resolution. The challenge is to segment 11 classes.

The encoder and decoder weights are initialized as explained in He et al. and in order to train all the variants, the stochastic grandient descent is used with a fixed learning rate of 0.1 and a momentum of 0.9. Before each epoch the training set is shuffled and the minibatched picked is 12 images. The crossentropy loss is used as objective function.

To compare the performance of the different decoder networks the performance measures used are global accuracy (G, percentage of pixels correctly classified), class average accuracy (C, mean of the predictive accuracy over all classes) and mean intersection over union (mIoU, stringent metric that penalizes the false positive prediction).

The key idea in computing a semantic contour score is to evaluate the F1-measure (computing Recall and Precision too). The F1 measure for each class is averaged by the F1 measure of each image in the class.

Each architecture is tested after 1000 iterations optimizations on the CamVid validation set until the training loss converges. It is important to achieve high global accuracies as is the best indicator for self-autonomous driving, as the major part of the pixels form part of the roads, buildings, etc.


## Results & Future Work
Bilinear interpolation based upsampling without any learning performs the worst based on the measures of accuracy. All the other methods (FNC-Basic or Segnet-Basic) perform better. 

When comparing the FNC with SegNet it can be seen that they perform equally in the datasets, the difference is that SegNet uses less memory during inference since it only stores max-pooling indices. 

SegNet Basic is most similar to FCN Basic-NoAddition in therms of decoder, however, Segnet has a better performance and is also larger than FCN. The accuracy of FCN Basic-NoAddition is also lower than the FCN Basic, showing that is vital to capture the information present in the encoder feature maps.

The size of FCN Basic-NoAddition-NoDimReduction model is slightly larger than the SegNet-Basic since the final enconder feature maps are not compressed to match the number of classes K. The performance of this FCN is poorer than the SegNet-Basic.

The comparison between FCN-BasicNoAddition and SegNet Basic-SingleChannelDecoder shows that using max-pooling indices for upsampling and an overall larger decoder leads to better perfomance. Also, the results show that when no class balancing is used, the results are poorer for all the variants, particularly ofr class average accuracy and mIoU metric. 

Sumarizing, the best performance is achieved when enconder feature maps are stored in full, when memory during inference is constrained and then compressed, it can be used to improve the performance and that larger decoders increase performance.

### Segnet Results
The performance of SegNet is quantified on two scenes segmentation benchmarking, one in road scene segmentation for autonomous driving and the other is indoor scene segmentation which is used for Augmented Reality. The input RGB images for both cases are 360 x 480. 

Comparing Segnet with the other deep architecturres such as FCN, DeepLab-LargFOV and DeconvNet the same learning rate, momentum and different parameters are set to the same values. 

The qualitative results show the ability of the proposed architecture to segment smaller classes in road scenes. SegNet shows superior performances as compared with the other architectures. SegNet obtains competitive results when compared with methods chich use CRFs, showing the ability of the architecture to extract meaningful features from the input image and map it, showing that there is a huge improvement in the mIOU parameter. 

Also, SegNet and DeepConvnet achieve the highest scores in all the metrics compared with the other models. DeconvNet has a higher boundary delineation accuracy but SegNet is more efficient. FCN trains more slowly than the previous two. 


### Future Work & Conclusions
Deep learning models have been able to increase the succes in image segmentation. However, factors like memory and computational time during training and testing are also increaded. Training time becomes an important consideration if the performance increase is not meaningful and the Test time and memory are important in order to deploy models on specialized devices. The principal motivation under SegNet was to achieve a good performance with less time and computation resources in order to advance in that field, being a more efficient architecture because it only stores the max-pooling indices. It performs competitively in well-known datasets, achieving very good results in image segmentation.



















